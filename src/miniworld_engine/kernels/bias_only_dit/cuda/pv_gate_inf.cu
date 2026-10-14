// Hopper prototype: same PV+gate, dP+softmax-backward, and expand+SwiGLU boundaries as B200.
// The B200 implementation below is retained unchanged.
#ifdef MINIWORLD_BO_SM90
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
#include "wgmma_n.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace bo90 {
using bf=__nv_bfloat16;
TMN_DEVI void mma64(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b));
}
TMN_DEVI void mma64mn(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b));
}
TMN_DEVI uint64_t ds(uint32_t p){ return smem_desc(p,16,1024,1); }
TMN_DEVI void tma_store2d(const CUtensorMap*m,const void*src,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"::"l"(m),"r"(smem_u32(src)),"r"(c0),"r"(c1):"memory");
}
TMN_DEVI void bulk_commit_(){asm volatile("cp.async.bulk.commit_group;":::"memory");}
template<int N> TMN_DEVI void bulk_wait_read_(){asm volatile("cp.async.bulk.wait_group.read %0;"::"n"(N):"memory");}
TMN_DEVI void bulk_wait_all_(){asm volatile("cp.async.bulk.wait_group 0;":::"memory");}
using Encode=CUresult (*)(CUtensorMap*,CUtensorMapDataType,cuuint32_t,void*,const cuuint64_t*,const cuuint64_t*,const cuuint32_t*,const cuuint32_t*,CUtensorMapInterleave,CUtensorMapSwizzle,CUtensorMapL2promotion,CUtensorMapFloatOOBfill);
CUtensorMap map(at::Tensor t,int bc,int br){
 static Encode enc=[] { void*p=nullptr; cudaDriverEntryPointQueryResult q{}; TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&p,cudaEnableDefault,&q)==cudaSuccess&&p,"TMA encoder"); return reinterpret_cast<Encode>(p); }();
 CUtensorMap m{}; const cuuint64_t dims[2]={(cuuint64_t)t.size(1),(cuuint64_t)t.size(0)},str[1]={(cuuint64_t)t.stride(0)*2}; const cuuint32_t box[2]={(unsigned)bc,(unsigned)br},e[2]={1,1};
 TORCH_CHECK(enc(&m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,t.data_ptr(),dims,str,box,e,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS,"TMA encode"); return m;
}
template<bool DPB,bool GATE>
__global__ __launch_bounds__(128,2) void core(const __grid_constant__ CUtensorMap ma,const __grid_constant__ CUtensorMap mb,
 const bf* p,const bf* g,const float* dd,bf*out,int L,int A,int H,int DH,int gs,int os) {
 __shared__ __align__(1024) unsigned char sm[24576];
 __shared__ uint64_t bar;
 int tid=threadIdx.x,warp=tid/32,lane=tid%32,mt=L/64;
 if(tid==0){mbar_init(&bar,1);fence_barrier_init();}
 __syncthreads();
 int nt=DPB?mt:A,items=H*mt*nt; unsigned phase=0;
 for(int item=blockIdx.x;item<items;item+=gridDim.x){
   int n=item%nt,head=(item/nt)/mt,m0=((item/nt)%mt)*64;
   float acc[32]={};
   int steps=DPB?A:L/64;
   for(int kb=0;kb<steps;++kb){
     if(tid==0){
       mbar_arrive_expect_tx(&bar,16384);
       if constexpr(DPB){tma_load_2d(sm,&ma,&bar,head*DH,kb*L+m0); tma_load_2d(sm+8192,&mb,&bar,head*DH,kb*L+n*64);}
       else {tma_load_2d(sm,&ma,&bar,kb*64,head*L+m0);tma_load_2d(sm+8192,&mb,&bar,head*DH,n*L+kb*64);}
     }
     mbar_wait(&bar,phase++&1); __syncthreads();
     if constexpr(!DPB){
       for(int e=tid;e<4096;e+=128){int r=e/64,c=e%64;*reinterpret_cast<bf*>(sm+16384+swz128((uint32_t)c,(uint32_t)(r*2)))=*reinterpret_cast<bf*>(sm+8192+swz128((uint32_t)r,(uint32_t)(c*2)));}
       fence_proxy_async(); __syncthreads();
     }
     fence_regs(acc);wgmma_fence();
     for(int k=0;k<(DPB?DH/16:4);++k)mma64(acc,ds(smem_u32(sm)+k*32),ds(smem_u32(sm)+(DPB?8192:16384)+k*32));
     wgmma_commit();wgmma_wait<0>();fence_regs(acc); __syncthreads();
   }
   int r0=16*warp+(lane>>2),c0=2*(lane&3);
   #pragma unroll
   for(int j=0;j<8;++j)for(int rb=0;rb<2;++rb){
     int r=m0+r0+rb*8,c=j*8+c0;
     #pragma unroll
     for(int q=0;q<2;++q){
       float v=acc[j*4+rb*2+q];
       if constexpr(DPB){float d=0;for(int a=0;a<A;++a)d+=dd[(a*H+head)*L+r];int ix=(head*L+r)*L+n*64+c+q;out[ix]=__float2bfloat16_rn(__bfloat162float(p[ix])*(v-d));}
       else if(c+q<DH){int row=n*L+r,col=head*DH+c+q;if constexpr(GATE)v/=1.f+__expf(-__bfloat162float(g[row*gs+col]));out[row*os+col]=__float2bfloat16_rn(v);}
     }
   }
   __syncthreads();
 }
}
template<bool GATE, bool DIRECT=false>
__global__ __launch_bounds__(256,1) void pv_grouped(const __grid_constant__ CUtensorMap mp,const __grid_constant__ CUtensorMap mv,
 const bf*g,bf*out,int L,int A,int H,int DH,int gs,int os,int SG) {
 const int RAW=DIRECT?L*128:98304,TRANS=RAW+3*8192,BARS=TRANS+(DIRECT?0:3*8192);
 extern __shared__ __align__(1024) unsigned char sm[];
 uint64_t*pb=reinterpret_cast<uint64_t*>(sm+BARS),*full=pb+1,*empty=full+3;
 int tid=threadIdx.x,wtid=tid&127,warp=wtid/32,lane=tid%32,mt=L/64,ng=(A+SG-1)/SG;
 if(!tid){mbar_init(pb,1);for(int s=0;s<3;++s){mbar_init(full+s,1);mbar_init(empty+s,1);}fence_barrier_init();}
 __syncthreads();unsigned it=0,pi=0;
 for(int item=blockIdx.x;item<H*mt*ng;item+=gridDim.x){
   int group=item%ng,head=item/(ng*mt),m0=((item/ng)%mt)*64,a0=group*SG,ns=min(SG,A-a0),KB=L/64;
   if(!tid){mbar_arrive_expect_tx(pb,L*128);for(int kb=0;kb<KB;++kb)tma_load_2d(sm+kb*8192,&mp,pb,kb*64,head*L+m0);}
   mbar_wait(pb,pi++&1);__syncthreads();
   if(tid<128){
     if(!tid)for(int a=0;a<ns;++a)for(int kb=0;kb<KB;++kb,++it){
       int s=it%3;if(it>=3)mbar_wait(empty+s,((it/3)-1)&1);
       mbar_arrive_expect_tx(full+s,8192);tma_load_2d(sm+RAW+s*8192,&mv,full+s,head*DH,(a0+a)*L+kb*64);
     }
   }else{
     for(int a=0;a<ns;++a){
       float acc[32]={};
       int prev=-1;
       for(int kb=0;kb<KB;++kb,++it){
         int s=it%3;mbar_wait(full+s,(it/3)&1);named_bar_sync(1,128);
         if constexpr(!DIRECT){
         for(int e=wtid;e<4096;e+=128){int r=e/64,c=e%64;*reinterpret_cast<bf*>(sm+TRANS+s*8192+swz128((uint32_t)c,(uint32_t)(r*2)))=*reinterpret_cast<bf*>(sm+RAW+s*8192+swz128((uint32_t)r,(uint32_t)(c*2)));}
         fence_proxy_async();named_bar_sync(1,128);
         }
         fence_regs(acc);wgmma_fence();
         #pragma unroll
         for(int k=0;k<4;++k){
           if constexpr(DIRECT)mma64mn(acc,ds(smem_u32(sm)+kb*8192+k*32),ds(smem_u32(sm)+RAW+s*8192+k*2048));
           else mma64(acc,ds(smem_u32(sm)+kb*8192+k*32),ds(smem_u32(sm)+TRANS+s*8192+k*32));
         }
         wgmma_commit();
         if constexpr(DIRECT){if(prev>=0){wgmma_wait<1>();fence_regs(acc);if(!wtid)mbar_arrive(empty+prev);}prev=s;}
         else {wgmma_wait<0>();fence_regs(acc);named_bar_sync(1,128);if(!wtid)mbar_arrive(empty+s);}
       }
       if constexpr(DIRECT){wgmma_wait<0>();fence_regs(acc);if(!wtid)mbar_arrive(empty+prev);}
       int r0=m0+16*warp+(lane>>2),c0=2*(lane&3);
       #pragma unroll
       for(int j=0;j<8;++j)for(int rb=0;rb<2;++rb)for(int q=0;q<2;++q){
         int c=j*8+c0+q;if(c<DH){int row=(a0+a)*L+r0+rb*8,col=head*DH+c;float v=acc[j*4+rb*2+q];if constexpr(GATE)v/=1.f+__expf(-__bfloat162float(g[row*gs+col]));out[row*os+col]=__float2bfloat16_rn(v);}
       }
     }
   }
   __syncthreads();
 }
}
void grouped_pv(at::Tensor v,at::Tensor p,at::Tensor out,int64_t A,c10::optional<at::Tensor> gate,int64_t H,int64_t DH,int64_t SG){
 at::cuda::CUDAGuard guard(v.device());cudaSetDevice(v.get_device());int L=v.size(0)/A;
 TORCH_CHECK(L%128==0&&L<=768&&SG>0,"grouped PV shape");constexpr int smem=98304+6*8192+1024;
 auto ma=map(p,64,64),mb=map(v,64,64);int grid=std::min<int>(132,H*(L/64)*((A+SG-1)/SG));
 if(gate){cudaFuncSetAttribute(pv_grouped<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);pv_grouped<true><<<grid,256,smem,at::cuda::getCurrentCUDAStream()>>>(ma,mb,(bf*)gate->data_ptr(),(bf*)out.data_ptr(),L,A,H,DH,gate->stride(0),out.stride(0),SG);}
 else{cudaFuncSetAttribute(pv_grouped<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);pv_grouped<false><<<grid,256,smem,at::cuda::getCurrentCUDAStream()>>>(ma,mb,nullptr,(bf*)out.data_ptr(),L,A,H,DH,0,out.stride(0),SG);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void grouped_pv_pipeline(at::Tensor v,at::Tensor p,at::Tensor out,int64_t A,c10::optional<at::Tensor> gate,int64_t H,int64_t DH,int64_t SG){
 at::cuda::CUDAGuard guard(v.device());cudaSetDevice(v.get_device());int L=v.size(0)/A;
 TORCH_CHECK(L%128==0&&L<=768&&SG>0,"grouped PV shape");int smem=L*128+3*8192+1024;
 auto ma=map(p,64,64),mb=map(v,64,64);int grid=std::min<int>(264,H*(L/64)*((A+SG-1)/SG));
 if(gate){cudaFuncSetAttribute(pv_grouped<true,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);pv_grouped<true,true><<<grid,256,smem,at::cuda::getCurrentCUDAStream()>>>(ma,mb,(bf*)gate->data_ptr(),(bf*)out.data_ptr(),L,A,H,DH,gate->stride(0),out.stride(0),SG);}
 else{cudaFuncSetAttribute(pv_grouped<false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);pv_grouped<false,true><<<grid,256,smem,at::cuda::getCurrentCUDAStream()>>>(ma,mb,nullptr,(bf*)out.data_ptr(),L,A,H,DH,0,out.stride(0),SG);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- pvx: P V with every sample of a group in ONE pass over the keys --------------------------------------------------
// CTA = (head, 64 * WGS rows, group of <= SGX samples); WGS consumer warpgroups own 64 rows each and keep one fp32 accumulator
// per sample, a producer thread streams [P rows | v of every sample] k-blocks through a TMA ring. P is read once per
// (head, rows, group); nothing is resident, so there is no serial prologue and the next item's loads overlap this one's
// epilogue. CS CTAs of a cluster work on consecutive row tiles of the same head and group: they need the same v blocks, so
// each loads (CS-th of) the samples and multicasts them to the whole cluster (v L2->SM bytes / CS). The host picks
// (WGS, SG, CS) by a cost model so that small L still fills the SMs.
constexpr int SGX=5;
// wgmma m64nNk16, B transposed (N-major), N = 64 x samples: the sample tiles sit 8 KB apart (descriptor LBO)
TMN_DEVI void mma_tn64(float*d,uint64_t a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b));
}
TMN_DEVI void mma_tn128(float*d,uint64_t a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b));
}
TMN_DEVI void mma_tn192(float*d,uint64_t a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, 1, 1, 1, 0, 1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]) : "l"(a), "l"(b));
}
TMN_DEVI uint64_t dsl(uint32_t p){ return smem_desc(p,8192,1024,1); }
template<int WGS> struct PvxCfg { static constexpr int ST=WGS==2?3:4, SB=(WGS+SGX)*8192; };
TMN_DEVI void tma_mc(void*dst,const CUtensorMap*m,uint64_t*bar,int c0,int c1,uint16_t mask){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
    :: "r"(smem_u32(dst)), "l"(m), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
TMN_DEVI uint32_t mapa_(uint32_t a,uint32_t rank){uint32_t r;asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(r):"r"(a),"r"(rank));return r;}
TMN_DEVI void mbar_arrive_remote_(uint64_t*bar,uint32_t rank){asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(mapa_(smem_u32(bar),rank)):"memory");}
TMN_DEVI void cluster_sync_(){asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n":::"memory");}
TMN_DEVI uint32_t cluster_rank_(){uint32_t r;asm volatile("mov.u32 %0, %%cluster_ctarank;\n":"=r"(r));return r;}
__device__ long long pvx_prof[512*8];
template<bool GATE,int WGS,int CS,bool PROF=false>
__global__ __launch_bounds__(128*(WGS+1),1) void pvx(const __grid_constant__ CUtensorMap mp,const __grid_constant__ CUtensorMap mv,const __grid_constant__ CUtensorMap mo,const __grid_constant__ CUtensorMap mg,
 const bf*g,bf*out,int L,int A,int H,int DH,int gs,int os,int SG) {
 constexpr int ST=PvxCfg<WGS>::ST,SB=PvxCfg<WGS>::SB;
 extern __shared__ __align__(1024) unsigned char sm[];
 uint64_t*full=reinterpret_cast<uint64_t*>(sm+ST*SB),*empty=full+ST,*gfull=empty+ST;
 constexpr int GB=64*48*2;                                       // one 64 x 48 bf16 tile (the TMA epilogue, DH == 48)
 unsigned char*gst=sm+ST*SB+256,*ost=gst+WGS*2*GB;
 const bool tmae=DH==48;
 int tid=threadIdx.x,wg=tid/128,wtid=tid&127,warp=wtid/32,lane=tid&31,mt=L/(64*WGS),ng=(A+SG-1)/SG,KB=L/64;
 const uint32_t rank=CS>1?cluster_rank_():0;
 if(!tid){for(int s=0;s<ST;++s){mbar_init(full+s,1);mbar_init(empty+s,WGS*CS);}for(int b=0;b<2*WGS;++b)mbar_init(gfull+b,1);fence_barrier_init();}
 __syncthreads();
 if(CS>1)cluster_sync_();
 const int mtc=mt/CS,citems=H*mtc*ng,cstride=gridDim.x/CS;
 unsigned it=0;
 if(wg==0){
   setmaxnreg_dec<40>();
   long long pw=0,pt=0;if(PROF)pt=clock64();
   if(tid==0)for(int ci=blockIdx.x/CS;ci<citems;ci+=cstride){
     int m=(ci%mtc)*CS+rank,head=(ci/mtc)%H,group=ci/(mtc*H),a0=group*SG,ns=min(SG,A-a0),m0=m*64*WGS;
     for(int kb=0;kb<KB;++kb,++it){
       long long t0=0;if(PROF)t0=clock64();
       int s=it%ST;if(it>=ST)mbar_wait(empty+s,((it/ST)-1)&1);
       if(PROF){pw+=clock64()-t0;}
       unsigned char*st=sm+s*SB;
       mbar_arrive_expect_tx(full+s,(WGS+ns)*8192);
       #pragma unroll
       for(int w=0;w<WGS;++w)tma_load_2d(st+w*8192,&mp,full+s,kb*64,head*L+m0+w*64);
       for(int j=0;j<ns;++j){
         if(CS==1)tma_load_2d(st+WGS*8192+j*8192,&mv,full+s,head*DH,(a0+j)*L+kb*64);
         else if((unsigned)(j%CS)==rank)tma_mc(st+WGS*8192+j*8192,&mv,full+s,head*DH,(a0+j)*L+kb*64,(uint16_t)((1u<<CS)-1));
       }
     }
   }
   if(PROF&&tid==0){pvx_prof[blockIdx.x*8+6]=pw;pvx_prof[blockIdx.x*8+7]=clock64()-pt;}
 }else{
 setmaxnreg_inc<232>();
 const int cw=wg-1;
 long long cw_full=0,cw_mma=0,cw_t0=0;if(PROF)cw_t0=clock64();
 uint32_t gph=0;
 for(int ci=blockIdx.x/CS;ci<citems;ci+=cstride){
   int m=(ci%mtc)*CS+rank,head=(ci/mtc)%H,group=ci/(mtc*H),a0=group*SG,ns=min(SG,A-a0),m0=m*64*WGS+cw*64;
   // chunk 0: samples 0..2 as ONE n192 / n128 / n64 wgmma; chunk 1: samples 3..4 as n128 / n64 (N = 64 x samples)
   const int r0=m0+16*warp+(lane>>2),c0=2*(lane&3);
   // the gate rows of the item's first sample: issued before the K loop, so their latency hides under the MMAs
   uint32_t gcur[16],gnxt[16];
   auto loadg=[&](int j,uint32_t (&dst)[16]){
     #pragma unroll
     for(int jj=0;jj<8;++jj)
       #pragma unroll
       for(int rb=0;rb<2;++rb){
         int c=jj*8+c0;uint32_t v=0;
         if(GATE&&c<DH&&j<ns)v=__ldg(reinterpret_cast<const unsigned*>(g+(size_t)((a0+j)*L+r0+rb*8)*gs+head*DH+c));
         dst[jj*2+rb]=v;
       }
   };
   if(tmae){
     if(GATE&&wtid==0){
       #pragma unroll
       for(int b=0;b<2;++b)if(b<ns){mbar_arrive_expect_tx(gfull+cw*2+b,GB);tma_load_2d(gst+(cw*2+b)*GB,&mg,gfull+cw*2+b,head*DH,(a0+b)*L+m0);}
     }
   }else loadg(0,gcur);
   float acc0[96],acc1[64];
   #pragma unroll
   for(int e=0;e<96;++e)acc0[e]=0.f;
   #pragma unroll
   for(int e=0;e<64;++e)acc1[e]=0.f;
   int prev=-1;
   for(int kb=0;kb<KB;++kb,++it){
     long long t0=0;if(PROF)t0=clock64();
     int s=it%ST;mbar_wait(full+s,(it/ST)&1);
     if(PROF)cw_full+=clock64()-t0;
     uint32_t a_base=smem_u32(sm)+s*SB+cw*8192,b_base=smem_u32(sm)+s*SB+WGS*8192;
     fence_regs(acc0);fence_regs(acc1);
     wgmma_fence();
     #pragma unroll
     for(int k=0;k<4;++k){
       uint64_t da=ds(a_base+k*32),db0=dsl(b_base+k*2048),db1=dsl(b_base+3*8192+k*2048);
       if(ns>=3)mma_tn192(acc0,da,db0);else if(ns==2)mma_tn128(acc0,da,db0);else mma_tn64(acc0,da,db0);
       if(ns==5)mma_tn128(acc1,da,db1);else if(ns==4)mma_tn64(acc1,da,db1);
     }
     wgmma_commit();
     if(prev>=0){
       long long t1=0;if(PROF)t1=clock64();
       wgmma_wait<1>();
       if(PROF)cw_mma+=clock64()-t1;
       if(!wtid){ if(CS==1)mbar_arrive(empty+prev); else for(int r=0;r<CS;++r)mbar_arrive_remote_(empty+prev,r); }
     }
     prev=s;
   }
   wgmma_wait<0>();
   fence_regs(acc0);fence_regs(acc1);
   if(!wtid){ if(CS==1)mbar_arrive(empty+prev); else for(int r=0;r<CS;++r)mbar_arrive_remote_(empty+prev,r); }
   if(tmae){
     // the gate tile arrives by TMA, the bf16 result is staged in shared memory and leaves as ONE TMA store per sample: no per-thread
     // global access in the epilogue (they were 30 % of the kernel's stall samples: the LSU queue)
     const int lr0=16*warp+(lane>>2);
     #pragma unroll
     for(int j=0;j<SGX;++j){
       if(j<ns){
         const int b=j&1;
         if(GATE){mbar_wait(gfull+cw*2+b,(gph>>b)&1);gph^=1u<<b;}
         if(wtid==0){ if(j==0)bulk_wait_read_<0>(); else bulk_wait_read_<1>(); }  // the store that used this staging buffer has left it
         named_bar_sync(1+cw,128);
         unsigned char*ob=ost+(cw*2+b)*GB;const unsigned char*gb=gst+(cw*2+b)*GB;
         #pragma unroll
         for(int jj=0;jj<6;++jj){
           #pragma unroll
           for(int rb=0;rb<2;++rb){
             const int lr=lr0+rb*8,off=lr*(48*2)+(jj*8+c0)*2;
             const int ix=((j%3)*8+jj)*4+rb*2;
             float v0=(j<3)?acc0[ix]:acc1[ix],v1=(j<3)?acc0[ix+1]:acc1[ix+1];
             if constexpr(GATE){
               const uint32_t gw=*reinterpret_cast<const uint32_t*>(gb+off);
               v0*=rcpf(1.f+ex2f(-1.4426950408889634f*__uint_as_float(gw<<16)));v1*=rcpf(1.f+ex2f(-1.4426950408889634f*__uint_as_float(gw&0xffff0000u)));
             }
             *reinterpret_cast<__nv_bfloat162*>(ob+off)=__floats2bfloat162_rn(v0,v1);
           }
         }
         fence_proxy_async();
         named_bar_sync(1+cw,128);
         if(wtid==0){
           tma_store2d(&mo,ob,head*DH,(a0+j)*L+m0);bulk_commit_();
           if(GATE&&j+2<ns){mbar_arrive_expect_tx(gfull+cw*2+b,GB);tma_load_2d(gst+(cw*2+b)*GB,&mg,gfull+cw*2+b,head*DH,(a0+j+2)*L+m0);}
         }
       }
     }
   }else{
   #pragma unroll
   for(int j=0;j<SGX;++j){
     if(j<ns){
       loadg(j+1,gnxt);
       #pragma unroll
       for(int jj=0;jj<8;++jj){
         int c=jj*8+c0;
         if(c<DH){
           #pragma unroll
           for(int rb=0;rb<2;++rb){
             int row=(a0+j)*L+r0+rb*8,col=head*DH+c;
             int ix=((j%3)*8+jj)*4+rb*2;
             float v0=(j<3)?acc0[ix]:acc1[ix],v1=(j<3)?acc0[ix+1]:acc1[ix+1];
             if constexpr(GATE){
               uint32_t gw=gcur[jj*2+rb];
               v0*=rcpf(1.f+ex2f(-1.4426950408889634f*__uint_as_float(gw<<16)));v1*=rcpf(1.f+ex2f(-1.4426950408889634f*__uint_as_float(gw&0xffff0000u)));
             }
             *reinterpret_cast<__nv_bfloat162*>(out+(size_t)row*os+col)=__floats2bfloat162_rn(v0,v1);
           }
         }
       }
       #pragma unroll
       for(int e=0;e<16;++e)gcur[e]=gnxt[e];
     }
   }
   }
 }
 if(tmae&&wtid==0)bulk_wait_all_();
 if(PROF&&!wtid){long long*q=pvx_prof+blockIdx.x*8+cw*3;q[0]=cw_full;q[1]=cw_mma;q[2]=clock64()-cw_t0;}
 }
 if(CS>1){__syncwarp();cluster_sync_();}  // no remote arrive / multicast may land on an exited CTA
}
// (WGS, SG, CS) by a cost model: rounds x per-item cost (k-blocks x max(tensor time, L2 bytes at ~30 B/clk) + fixed).
static void pvx_pick(int L,int A,int H,int nsm,int&wgs,int&sg,int&cs,int fw=0,int fg=0,int fc=0){
 double best=1e30;int KB=L/64;
 for(int w=1;w<=2;++w){ if(fw&&w!=fw)continue; if(L%(64*w))continue; int mt=L/(64*w);
  for(int c:{1,2,3,6}){ if(fc?c!=fc:c!=1)continue; if(mt%c)continue;
   for(int g=1;g<=SGX;++g){ if(fg&&g!=fg)continue;
     int ng=(A+g-1)/g, citems=H*(mt/c)*ng, cl=std::max(1,std::min(citems,nsm/c)), rounds=(citems+cl-1)/cl;
     double kcost=std::max(128.0*g*w,(double)(w+(double)g/c)*8192/30.0);
     double t=rounds*(KB*kcost+3000.0)+(c>1?2000.0:0.0);
     if(t<best){best=t;wgs=w;sg=g;cs=c;}
   }}}
}
static CUtensorMap map_lin(at::Tensor t,int bc,int br){
 static Encode enc=[] { void*p=nullptr; cudaDriverEntryPointQueryResult q{}; TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&p,cudaEnableDefault,&q)==cudaSuccess&&p,"TMA encoder"); return reinterpret_cast<Encode>(p); }();
 CUtensorMap m{}; const cuuint64_t dims[2]={(cuuint64_t)t.size(1),(cuuint64_t)t.size(0)},str[1]={(cuuint64_t)t.stride(0)*2}; const cuuint32_t box[2]={(unsigned)bc,(unsigned)br},e[2]={1,1};
 TORCH_CHECK(enc(&m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,t.data_ptr(),dims,str,box,e,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS,"TMA encode linear"); return m;
}
template<int WGS> constexpr int pvx_smem(){ return PvxCfg<WGS>::ST*PvxCfg<WGS>::SB+256+WGS*2*6144*2+1024; }
template<bool GATE,int WGS,int CS> static void pvx_launch(at::Tensor v,at::Tensor p,at::Tensor out,int A,c10::optional<at::Tensor> gate,int H,int DH,int nsm,int SG){
 int L=v.size(0)/A;constexpr int smem=pvx_smem<WGS>();
 auto ma=map(p,64,64),mb=map(v,64,64);int ng=(A+SG-1)/SG;int mt=L/(64*WGS);int citems=H*(mt/CS)*ng;
 auto mo=map_lin(out,(int)DH,64),mg=GATE?map_lin(*gate,(int)DH,64):map_lin(out,(int)DH,64);
 int grid=std::max(1,std::min(citems,nsm/CS))*CS;
 auto kern=pvx<GATE,WGS,CS>;
 cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(128*(WGS+1));cfg.dynamicSmemBytes=smem;cfg.stream=at::cuda::getCurrentCUDAStream();
 cudaLaunchAttribute at[1];at[0].id=cudaLaunchAttributeClusterDimension;at[0].val.clusterDim.x=CS;at[0].val.clusterDim.y=1;at[0].val.clusterDim.z=1;
 cfg.attrs=at;cfg.numAttrs=1;
 TORCH_CHECK(cudaLaunchKernelEx(&cfg,kern,ma,mb,mo,mg,GATE?(const bf*)gate->data_ptr():(const bf*)nullptr,(bf*)out.data_ptr(),L,A,H,DH,GATE?(int)gate->stride(0):0,(int)out.stride(0),SG)==cudaSuccess,"pvx launch failed");
}
template<bool GATE,int WGS> static void pvx_dispatch(int cs,at::Tensor v,at::Tensor p,at::Tensor out,int A,c10::optional<at::Tensor> gate,int H,int DH,int nsm,int sg){
 switch(cs){
  case 1: pvx_launch<GATE,WGS,1>(v,p,out,A,gate,H,DH,nsm,sg);break;
  case 2: pvx_launch<GATE,WGS,2>(v,p,out,A,gate,H,DH,nsm,sg);break;
  case 3: pvx_launch<GATE,WGS,3>(v,p,out,A,gate,H,DH,nsm,sg);break;
  default: pvx_launch<GATE,WGS,6>(v,p,out,A,gate,H,DH,nsm,sg);break;
 }
}
at::Tensor pv_cta_prof(at::Tensor v,at::Tensor p,at::Tensor out,int64_t A,at::Tensor gate,int64_t H,int64_t DH,int64_t nsm){
 at::cuda::CUDAGuard guard(v.device());cudaSetDevice(v.get_device());int L=v.size(0)/A;
 int ng=(A+SGX-1)/SGX,SG=(A+ng-1)/ng;constexpr int WGS=2;constexpr int smem=pvx_smem<WGS>();
 auto ma=map(p,64,64),mb=map(v,64,64);auto mo=map_lin(out,(int)DH,64),mg=map_lin(gate,(int)DH,64);int mt=L/(64*WGS);int citems=H*mt*((A+SG-1)/SG);int grid=std::max(1,std::min<int>(citems,(int)nsm));
 auto kern=pvx<true,WGS,1,true>;cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 kern<<<grid,128*(WGS+1),smem,at::cuda::getCurrentCUDAStream()>>>(ma,mb,mo,mg,(bf*)gate.data_ptr(),(bf*)out.data_ptr(),L,A,H,DH,(int)gate.stride(0),(int)out.stride(0),SG);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 auto t=at::zeros({512,8},v.options().dtype(at::kLong));
 cudaDeviceSynchronize();cudaMemcpyFromSymbol(t.data_ptr(),pvx_prof,512*8*8,0,cudaMemcpyDeviceToDevice);
 return t.slice(0,0,grid);
}
void pv_cta(at::Tensor v,at::Tensor p,at::Tensor out,int64_t A,c10::optional<at::Tensor> gate,int64_t H,int64_t DH,int64_t nsm,int64_t fw,int64_t fg,int64_t fc){
 at::cuda::CUDAGuard guard(v.device());cudaSetDevice(v.get_device());int L=v.size(0)/A;
 TORCH_CHECK(L%128==0&&L<=768&&DH%2==0&&DH<=64&&v.scalar_type()==at::kBFloat16,"pvx shape");
 int wgs=2,sg=1,cs=1;pvx_pick(L,A,H,nsm,wgs,sg,cs,fw,fg,fc);
 if(gate){ if(wgs==2)pvx_dispatch<true,2>(cs,v,p,out,A,gate,H,DH,nsm,sg); else pvx_dispatch<true,1>(cs,v,p,out,A,gate,H,DH,nsm,sg); }
 else    { if(wgs==2)pvx_dispatch<false,2>(cs,v,p,out,A,gate,H,DH,nsm,sg); else pvx_dispatch<false,1>(cs,v,p,out,A,gate,H,DH,nsm,sg); }
}
// ---- fp32 (TF32) PV core: K-major A = P fp32 tile, K-major B = v^T rows of 48 per sample (N = 48 x samples) -----------
TMN_DEVI void mma_tf32_n48(float*d,uint64_t a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n48k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, 1, 1, 1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]) : "l"(a), "l"(b));
}
TMN_DEVI void mma_tf32_n96(float*d,uint64_t a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n96k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, 1, 1, 1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]) : "l"(a), "l"(b));
}
TMN_DEVI void mma_tf32_n144(float*d,uint64_t a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n144k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, %72, %73, 1, 1, 1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]) : "l"(a), "l"(b));
}
TMN_DEVI void mma_tf32_n192(float*d,uint64_t a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n192k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, 1, 1, 1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]) : "l"(a), "l"(b));
}
TMN_DEVI void mma_tf32_n240(float*d,uint64_t a,uint64_t b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n240k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119}, %120, %121, 1, 1, 1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]) : "l"(a), "l"(b));
}

CUtensorMap map32(at::Tensor t,int bc,int br){
 static Encode enc=[] { void*p=nullptr; cudaDriverEntryPointQueryResult q{}; TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&p,cudaEnableDefault,&q)==cudaSuccess&&p,"TMA encoder"); return reinterpret_cast<Encode>(p); }();
 CUtensorMap m{}; const cuuint64_t dims[2]={(cuuint64_t)t.size(1),(cuuint64_t)t.size(0)},str[1]={(cuuint64_t)t.stride(0)*4}; const cuuint32_t box[2]={(unsigned)bc,(unsigned)br},e[2]={1,1};
 TORCH_CHECK(enc(&m,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,t.data_ptr(),dims,str,box,e,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS,"TMA encode f32"); return m;
}
// pvx32: out[s] = sigmoid(g[s]) * (P_h v[s]_h) in fp32 on TF32 tensor cores. v^T [da, A L] fp32 (rows = attention channels,
// columns = tokens), P [H L, L] fp32, g / out [A L, da] fp32. CTA = (head, 64 * WGS rows, group of <= SGX samples); one
// m64n(48 x samples)k8 wgmma per k8 step covers the whole group; k-blocks of 32 keys through a TMA ring.
template<int WGS> struct Pvx32Cfg { static constexpr int ST=WGS==2?4:5, SB=WGS*8192+SGX*6144; };
template<int WGS>
__global__ __launch_bounds__(128*(WGS+1),1) void pvx32(const __grid_constant__ CUtensorMap mp,const __grid_constant__ CUtensorMap mv,
 const float*__restrict__ g,float*__restrict__ out,int L,int A,int H,int DH,int gs,int os,int SG) {
 constexpr int ST=Pvx32Cfg<WGS>::ST,SB=Pvx32Cfg<WGS>::SB;
 extern __shared__ __align__(1024) unsigned char sm[];
 uint64_t*full=reinterpret_cast<uint64_t*>(sm+ST*SB),*empty=full+ST;
 int tid=threadIdx.x,wg=tid/128,wtid=tid&127,warp=wtid/32,lane=tid&31,mt=L/(64*WGS),ng=(A+SG-1)/SG,KB=L/32;
 if(!tid){for(int s=0;s<ST;++s){mbar_init(full+s,1);mbar_init(empty+s,WGS);}fence_barrier_init();}
 __syncthreads();
 const int items=H*mt*ng;
 unsigned it=0;
 if(wg==0){
   setmaxnreg_dec<40>();
   if(tid==0)for(int item=blockIdx.x;item<items;item+=gridDim.x){
     int m=item%mt,head=(item/mt)%H,group=item/(mt*H),a0=group*SG,ns=min(SG,A-a0),m0=m*64*WGS;
     for(int kb=0;kb<KB;++kb,++it){
       int s=it%ST;if(it>=ST)mbar_wait(empty+s,((it/ST)-1)&1);
       unsigned char*st=sm+s*SB;
       mbar_arrive_expect_tx(full+s,WGS*8192+ns*6144);
       #pragma unroll
       for(int w=0;w<WGS;++w)tma_load_2d(st+w*8192,&mp,full+s,kb*32,head*L+m0+w*64);
       for(int j=0;j<ns;++j)tma_load_2d(st+WGS*8192+j*6144,&mv,full+s,(a0+j)*L+kb*32,head*DH);
     }
   }
   return;
 }
 setmaxnreg_inc<232>();
 const int cw=wg-1;
 for(int item=blockIdx.x;item<items;item+=gridDim.x){
   int m=item%mt,head=(item/mt)%H,group=item/(mt*H),a0=group*SG,ns=min(SG,A-a0),m0=m*64*WGS+cw*64;
   float acc[120];
   #pragma unroll
   for(int e=0;e<120;++e)acc[e]=0.f;
   int prev=-1;
   for(int kb=0;kb<KB;++kb,++it){
     int s=it%ST;mbar_wait(full+s,(it/ST)&1);
     uint32_t a_base=smem_u32(sm)+s*SB+cw*8192,b_base=smem_u32(sm)+s*SB+WGS*8192;
     fence_regs(acc);
     wgmma_fence();
     #pragma unroll
     for(int k=0;k<4;++k){
       uint64_t da=ds(a_base+k*32),db=ds(b_base+k*32);
       switch(ns){
         case 1: mma_tf32_n48(acc,da,db);break;
         case 2: mma_tf32_n96(acc,da,db);break;
         case 3: mma_tf32_n144(acc,da,db);break;
         case 4: mma_tf32_n192(acc,da,db);break;
         default: mma_tf32_n240(acc,da,db);break;
       }
     }
     wgmma_commit();
     if(prev>=0){wgmma_wait<1>();if(!wtid)mbar_arrive(empty+prev);}
     prev=s;
   }
   wgmma_wait<0>();
   fence_regs(acc);
   if(!wtid)mbar_arrive(empty+prev);
   int r0=m0+16*warp+(lane>>2),c0=2*(lane&3);
   float2 gcur[12],gnxt[12];
   auto loadg=[&](int j,float2 (&dst)[12]){
     #pragma unroll
     for(int jj=0;jj<6;++jj)
       #pragma unroll
       for(int rb=0;rb<2;++rb){
         float2 v=make_float2(0.f,0.f);
         if(j<ns)v=__ldg(reinterpret_cast<const float2*>(g+(size_t)((a0+j)*L+r0+rb*8)*gs+head*DH+jj*8+c0));
         dst[jj*2+rb]=v;
       }
   };
   loadg(0,gcur);
   #pragma unroll
   for(int j=0;j<SGX;++j){
     if(j<ns){
       loadg(j+1,gnxt);
       #pragma unroll
       for(int jj=0;jj<6;++jj)
         #pragma unroll
         for(int rb=0;rb<2;++rb){
           int row=(a0+j)*L+r0+rb*8,col=head*DH+jj*8+c0;
           int ix=(j*6+jj)*4+rb*2;
           float2 gg=gcur[jj*2+rb];
           float v0=acc[ix]*rcpf(1.f+ex2f(-1.4426950408889634f*gg.x));
           float v1=acc[ix+1]*rcpf(1.f+ex2f(-1.4426950408889634f*gg.y));
           *reinterpret_cast<float2*>(out+(size_t)row*os+col)=make_float2(v0,v1);
         }
       #pragma unroll
       for(int e=0;e<12;++e)gcur[e]=gnxt[e];
     }
   }
 }
}
static void pvx32_pick(int L,int A,int H,int nsm,int&wgs,int&sg){
 double best=1e30;int KB=L/32;
 for(int w=1;w<=2;++w){ if(L%(64*w))continue;
   for(int g=1;g<=SGX;++g){
     int ng=(A+g-1)/g, items=H*(L/(64*w))*ng, grid=std::min(items,nsm), rounds=(items+grid-1)/grid;
     double kcost=std::max(96.0*g*w,(double)(w*8192+g*6144)/30.0);
     double t=rounds*(KB*kcost+3000.0);
     if(t<best){best=t;wgs=w;sg=g;}
   }}
}
template<int WGS> static void pvx32_launch(at::Tensor vt,at::Tensor p,at::Tensor out,at::Tensor gate,int A,int H,int DH,int nsm,int SG){
 int L=vt.size(1)/A;constexpr int smem=Pvx32Cfg<WGS>::ST*Pvx32Cfg<WGS>::SB+64+1024;
 auto ma=map32(p,32,64),mb=map32(vt,32,DH);int ng=(A+SG-1)/SG;int grid=std::min<int>(nsm,H*(L/(64*WGS))*ng);
 cudaFuncSetAttribute(pvx32<WGS>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 pvx32<WGS><<<grid,128*(WGS+1),smem,at::cuda::getCurrentCUDAStream()>>>(ma,mb,(const float*)gate.data_ptr(),(float*)out.data_ptr(),L,A,H,DH,(int)gate.stride(0),(int)out.stride(0),SG);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void pv_cta_tf32(at::Tensor vt,at::Tensor p,at::Tensor out,int64_t A,at::Tensor gate,int64_t H,int64_t DH,int64_t nsm){
 at::cuda::CUDAGuard guard(vt.device());cudaSetDevice(vt.get_device());int L=vt.size(1)/A;
 TORCH_CHECK(L%128==0&&L<=768&&DH==48&&vt.scalar_type()==at::kFloat&&p.scalar_type()==at::kFloat&&gate.scalar_type()==at::kFloat&&out.scalar_type()==at::kFloat,"pvx32 shape/dtype");
 TORCH_CHECK(vt.size(0)==H*DH&&vt.is_contiguous()&&p.is_contiguous()&&gate.stride(1)==1&&out.stride(1)==1,"pvx32 layout");
 int wgs=2,sg=1;pvx32_pick(L,A,H,nsm,wgs,sg);
 if(wgs==2)pvx32_launch<2>(vt,p,out,gate,A,H,DH,nsm,sg);else pvx32_launch<1>(vt,p,out,gate,A,H,DH,nsm,sg);
}
__global__ __launch_bounds__(256,2) void dpb_pipe(const __grid_constant__ CUtensorMap md,const __grid_constant__ CUtensorMap mv,
 const bf*p,const float*dd,bf*out,int L,int A,int H,int DH){
 extern __shared__ __align__(1024) unsigned char sm[];uint64_t*full=reinterpret_cast<uint64_t*>(sm+3*16384),*empty=full+3;
 int tid=threadIdx.x,wtid=tid&127,warp=wtid/32,lane=tid%32,mt=L/64;
 if(!tid){for(int s=0;s<3;++s){mbar_init(full+s,1);mbar_init(empty+s,1);}fence_barrier_init();}
 __syncthreads();unsigned it=0;
 if(tid<128){
   if(!tid)for(int item=blockIdx.x;item<H*mt*mt;item+=gridDim.x){
     int n0=(item%mt)*64,m0=((item/mt)%mt)*64,head=item/(mt*mt);
     for(int a=0;a<A;++a,++it){int s=it%3;if(it>=3)mbar_wait(empty+s,((it/3)-1)&1);mbar_arrive_expect_tx(full+s,16384);tma_load_2d(sm+s*16384,&md,full+s,head*DH,a*L+m0);tma_load_2d(sm+s*16384+8192,&mv,full+s,head*DH,a*L+n0);}
   }
   return;
 }
 for(int item=blockIdx.x;item<H*mt*mt;item+=gridDim.x){
   int n0=(item%mt)*64,m0=((item/mt)%mt)*64,head=item/(mt*mt),prev=-1;float acc[32]={};
   for(int a=0;a<A;++a,++it){
     int s=it%3;mbar_wait(full+s,(it/3)&1);fence_regs(acc);wgmma_fence();
     for(int k=0;k<DH/16;++k)mma64(acc,ds(smem_u32(sm)+s*16384+k*32),ds(smem_u32(sm)+s*16384+8192+k*32));
     wgmma_commit();if(prev>=0){wgmma_wait<1>();fence_regs(acc);if(!wtid)mbar_arrive(empty+prev);}prev=s;
   }
   wgmma_wait<0>();fence_regs(acc);if(!wtid)mbar_arrive(empty+prev);
   int r0=m0+16*warp+(lane>>2),c0=n0+2*(lane&3);float d0=0,d1=0;
   for(int a=0;a<A;++a){d0+=dd[(a*H+head)*L+r0];d1+=dd[(a*H+head)*L+r0+8];}
   #pragma unroll
   for(int j=0;j<8;++j)for(int rb=0;rb<2;++rb){
     int e=j*4+rb*2,ix=(head*L+r0+rb*8)*L+c0+8*j;float d=rb?d1:d0;
     *reinterpret_cast<__nv_bfloat162*>(out+ix)=__floats2bfloat162_rn(__bfloat162float(p[ix])*(acc[e]-d),__bfloat162float(p[ix+1])*(acc[e+1]-d));
   }
 }
}

// ---- dpb2: dbias = P o (sum_a dO_a v_a^T - D), a 128 x TN output tile per CTA (was 64 x 64: ~1.8 GB of operand re-reads) --------
// Two consumer warpgroups own 64 rows each (one m64nTNk16 chain over the samples), a producer thread streams
// [dO rows | v keys] per sample through a TMA ring; dsum [H, L] = sum_a D is precomputed. Both operands are K-major (channels).
template<int TN> struct Dpb2Cfg { static constexpr int ST=TN==256?3:4, SB=(128+TN)*128; };   // + the epilogue's P / out chunks (64 KB)
template<int TN>
__global__ __launch_bounds__(384,1) void dpb2(const __grid_constant__ CUtensorMap md,const __grid_constant__ CUtensorMap mv,
 const __grid_constant__ CUtensorMap mpp,const __grid_constant__ CUtensorMap mout,
 const bf*__restrict__ p,const float*__restrict__ dsum,bf*__restrict__ out,int L,int A,int H,int DH){
 constexpr int ST=Dpb2Cfg<TN>::ST,SB=Dpb2Cfg<TN>::SB;
 extern __shared__ __align__(1024) unsigned char sm[];
 uint64_t*full=reinterpret_cast<uint64_t*>(sm+ST*SB),*empty=full+ST,*pfull=empty+ST;
 // per consumer warpgroup: two 64 x 64 chunks of P (TMA in) and two of the result (TMA out), 128-B swizzled, 8 KB each
 unsigned char*pst=sm+ST*SB+1024,*ost=pst+2*2*8192;
 int tid=threadIdx.x,wg=tid/128,wtid=tid&127,warp=wtid/32,lane=tid&31,mtl=L/128,ntl=L/TN;
 if(!tid){for(int s=0;s<ST;++s){mbar_init(full+s,1);mbar_init(empty+s,2);}for(int b=0;b<4;++b)mbar_init(pfull+b,1);fence_barrier_init();}
 __syncthreads();
 const int items=H*mtl*ntl;unsigned it=0;
 if(wg==0){
   setmaxnreg_dec<40>();
   if(tid==0)for(int item=blockIdx.x;item<items;item+=gridDim.x){
     int nt=item%ntl,mt=(item/ntl)%mtl,head=item/(ntl*mtl),m0=mt*128,n0=nt*TN;
     for(int a=0;a<A;++a,++it){
       int s=it%ST;if(it>=ST)mbar_wait(empty+s,((it/ST)-1)&1);
       unsigned char*st=sm+s*SB;
       mbar_arrive_expect_tx(full+s,SB);
       tma_load_2d(st,&md,full+s,head*DH,a*L+m0);
       tma_load_2d(st+8192,&md,full+s,head*DH,a*L+m0+64);
       #pragma unroll
       for(int j=0;j<TN/64;++j)tma_load_2d(st+16384+j*8192,&mv,full+s,head*DH,a*L+n0+j*64);
     }
   }
   return;
 }
 setmaxnreg_inc<232>();
 const int cw=wg-1;
 uint32_t pph=0;   // parity of the P-chunk barriers: persists across items (a barrier completes TN / 128 times per item)
 for(int item=blockIdx.x;item<items;item+=gridDim.x){
   int nt=item%ntl,mt=(item/ntl)%mtl,head=item/(ntl*mtl),n0=nt*TN,m0=mt*128+cw*64;
   float acc[TN/2];
   #pragma unroll
   for(int e=0;e<TN/2;++e)acc[e]=0.f;
   int prev=-1;
   for(int a=0;a<A;++a,++it){
     int s=it%ST;mbar_wait(full+s,(it/ST)&1);
     uint32_t a_base=smem_u32(sm)+s*SB+cw*8192,b_base=smem_u32(sm)+s*SB+16384;
     fence_regs(acc);wgmma_fence();
     #pragma unroll
     for(int k=0;k<3;++k){
       if constexpr(TN==256)mma_n256(acc,ds(a_base+k*32),ds(b_base+k*32));
       else mma_n128(acc,ds(a_base+k*32),ds(b_base+k*32));
     }
     wgmma_commit();
     if(prev>=0){wgmma_wait<1>();if(!wtid)mbar_arrive(empty+prev);}
     prev=s;
   }
   wgmma_wait<0>();fence_regs(acc);if(!wtid)mbar_arrive(empty+prev);
   const int lr0=16*warp+(lane>>2),t4=4*(lane&3);
   const float d0=dsum[head*L+m0+lr0],d1=dsum[head*L+m0+lr0+8];
   // the epilogue works on 64-column chunks: P in by TMA (the next chunk while this one is computed), the result staged in shared
   // memory and out by one TMA store per chunk -- no per-thread global access
   unsigned char*pb=pst+cw*2*8192,*ob=ost+cw*2*8192;
   if(wtid==0){
     #pragma unroll
     for(int q=0;q<2;++q){mbar_arrive_expect_tx(pfull+cw*2+q,8192);tma_load_2d(pb+q*8192,&mpp,pfull+cw*2+q,n0+q*64,head*L+m0);}
   }
   #pragma unroll
   for(int q=0;q<TN/64;++q){
     const int b=q&1;
     mbar_wait(pfull+cw*2+b,(pph>>b)&1);pph^=1u<<b;
     if(wtid==0){ if(q==0)bulk_wait_read_<0>(); else bulk_wait_read_<1>(); }
     named_bar_sync(1+cw,128);
     #pragma unroll
     for(int jj=0;jj<8;++jj){
       #pragma unroll
       for(int rb=0;rb<2;++rb){
         const int lr=lr0+rb*8;
         const int off=lr*128+((jj^(lr&7))<<4)+t4;               // 128-B swizzled (row, 16-B chunk jj, word t)
         const uint32_t pw=*reinterpret_cast<const uint32_t*>(pb+b*8192+off);
         const float d=rb?d1:d0;
         const int ix=(q*8+jj)*4+rb*2;
         *reinterpret_cast<__nv_bfloat162*>(ob+b*8192+off)=__floats2bfloat162_rn(__uint_as_float(pw<<16)*(acc[ix]-d),__uint_as_float(pw&0xffff0000u)*(acc[ix+1]-d));
       }
     }
     fence_proxy_async();
     named_bar_sync(1+cw,128);
     if(wtid==0){
       tma_store2d(&mout,ob+b*8192,n0+q*64,head*L+m0);bulk_commit_();
       if(q+2<TN/64){mbar_arrive_expect_tx(pfull+cw*2+b,8192);tma_load_2d(pb+b*8192,&mpp,pfull+cw*2+b,n0+(q+2)*64,head*L+m0);}
     }
   }
 }
 if(wtid==0)bulk_wait_all_();
}
void dpb2_launch(at::Tensor d,at::Tensor v,at::Tensor p,at::Tensor dsum,at::Tensor out,int64_t A,int64_t H,int64_t DH,int64_t nsm){
 at::cuda::CUDAGuard guard(d.device());cudaSetDevice(d.get_device());int L=d.size(0)/A;
 TORCH_CHECK(L%128==0&&L<=768&&DH==48&&d.scalar_type()==at::kBFloat16&&dsum.scalar_type()==at::kFloat,"dpb2 shape/dtype");
 auto md=map(d,64,64),mv=map(v,64,64),mpp=map(p,64,64),mout=map(out,64,64);
 if(L%256==0){ constexpr int TN=256,smem=Dpb2Cfg<TN>::ST*Dpb2Cfg<TN>::SB+1024+8*8192+64+1024;
   cudaFuncSetAttribute(dpb2<TN>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
   int grid=std::min<int>(nsm,H*(L/128)*(L/TN));
   dpb2<TN><<<grid,384,smem,at::cuda::getCurrentCUDAStream()>>>(md,mv,mpp,mout,(bf*)p.data_ptr(),dsum.data_ptr<float>(),(bf*)out.data_ptr(),L,A,H,DH);
 } else { constexpr int TN=128,smem=Dpb2Cfg<TN>::ST*Dpb2Cfg<TN>::SB+1024+8*8192+64+1024;
   cudaFuncSetAttribute(dpb2<TN>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
   int grid=std::min<int>(nsm,H*(L/128)*(L/TN));
   dpb2<TN><<<grid,384,smem,at::cuda::getCurrentCUDAStream()>>>(md,mv,mpp,mout,(bf*)p.data_ptr(),dsum.data_ptr<float>(),(bf*)out.data_ptr(),L,A,H,DH);
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- dswi: dab = SwiGLU backward of (dz Wsq) in the GEMM's epilogue; dh never reaches HBM -----------------------------------------
// dh [M, H] = dz [M, K] Wsq [K, H] (B given K-major: WsqT [H, K]); with ab = [a | b] [M, 2H]: da = dh b silu'(a), db = dh silu(a).
// CTA tile 128 x 128 (two consumer warpgroups, m64n128), a TMA ring of [dz rows | WsqT rows] k-blocks; the tile's a / b
// fragments are loaded into registers before the K loop (their latency hides under the GEMM).
constexpr int DS_ST=4, DS_SB=32768;
__global__ __launch_bounds__(384,1) void dswi(const __grid_constant__ CUtensorMap mz,const __grid_constant__ CUtensorMap mw,
 const bf*__restrict__ ab,bf*__restrict__ dab,int M,int K,int H){
 extern __shared__ __align__(1024) unsigned char sm[];
 uint64_t*full=reinterpret_cast<uint64_t*>(sm+DS_ST*DS_SB),*empty=full+DS_ST;
 int tid=threadIdx.x,wg=tid/128,wtid=tid&127,warp=wtid/32,lane=tid&31,KB=K/64,MT=M/128,NT=H/128,items=MT*NT;
 if(!tid){for(int s=0;s<DS_ST;++s){mbar_init(full+s,1);mbar_init(empty+s,2);}fence_barrier_init();}
 __syncthreads();
 unsigned it=0;
 if(wg==0){
   setmaxnreg_dec<40>();
   if(tid==0)for(int item=blockIdx.x;item<items;item+=gridDim.x){
     int m0=(item%MT)*128,n0=(item/MT)*128;
     for(int kb=0;kb<KB;++kb,++it){
       int s=it%DS_ST;if(it>=DS_ST)mbar_wait(empty+s,((it/DS_ST)-1)&1);
       unsigned char*st=sm+s*DS_SB;
       mbar_arrive_expect_tx(full+s,DS_SB);
       tma_load_2d(st,&mz,full+s,kb*64,m0);
       tma_load_2d(st+8192,&mz,full+s,kb*64,m0+64);
       tma_load_2d(st+16384,&mw,full+s,kb*64,n0);
       tma_load_2d(st+24576,&mw,full+s,kb*64,n0+64);
     }
   }
   return;
 }
 setmaxnreg_inc<232>();
 const int cw=wg-1;
 for(int item=blockIdx.x;item<items;item+=gridDim.x){
   int m0=(item%MT)*128+cw*64,n0=(item/MT)*128;
   const int r0=m0+16*warp+(lane>>2),c0=n0+2*(lane&3);
   uint32_t av[32],bv[32];   // a, b bf16 pairs: [group j][row half rb]
   #pragma unroll
   for(int j=0;j<16;++j)
     #pragma unroll
     for(int rb=0;rb<2;++rb){
       size_t ix=(size_t)(r0+rb*8)*(2*H)+c0+8*j;
       av[j*2+rb]=*reinterpret_cast<const uint32_t*>(ab+ix);
       bv[j*2+rb]=*reinterpret_cast<const uint32_t*>(ab+ix+H);
     }
   float acc[64];
   #pragma unroll
   for(int e=0;e<64;++e)acc[e]=0.f;
   int prev=-1;
   for(int kb=0;kb<KB;++kb,++it){
     int s=it%DS_ST;mbar_wait(full+s,(it/DS_ST)&1);
     uint32_t a_base=smem_u32(sm)+s*DS_SB+cw*8192,b_base=smem_u32(sm)+s*DS_SB+16384;
     fence_regs(acc);wgmma_fence();
     #pragma unroll
     for(int k=0;k<4;++k)mma_n128(acc,ds(a_base+k*32),ds(b_base+k*32));
     wgmma_commit();
     if(prev>=0){wgmma_wait<1>();if(!wtid)mbar_arrive(empty+prev);}
     prev=s;
   }
   wgmma_wait<0>();fence_regs(acc);if(!wtid)mbar_arrive(empty+prev);
   #pragma unroll
   for(int j=0;j<16;++j)
     #pragma unroll
     for(int rb=0;rb<2;++rb){
       uint32_t aw=av[j*2+rb],bw=bv[j*2+rb];
       float a0=__uint_as_float(aw<<16),a1=__uint_as_float(aw&0xffff0000u),b0=__uint_as_float(bw<<16),b1=__uint_as_float(bw&0xffff0000u);
       float d0=acc[j*4+rb*2],d1=acc[j*4+rb*2+1];
       float s0=rcpf(1.f+ex2f(-1.4426950408889634f*a0)),s1=rcpf(1.f+ex2f(-1.4426950408889634f*a1));
       float da0=d0*b0*s0*(1.f+a0*(1.f-s0)),da1=d1*b1*s1*(1.f+a1*(1.f-s1));
       float db0=d0*a0*s0,db1=d1*a1*s1;
       size_t ix=(size_t)(r0+rb*8)*(2*H)+c0+8*j;
       *reinterpret_cast<__nv_bfloat162*>(dab+ix)=__floats2bfloat162_rn(da0,da1);
       *reinterpret_cast<__nv_bfloat162*>(dab+ix+H)=__floats2bfloat162_rn(db0,db1);
     }
 }
}
void swiglu_bwd_gemm(at::Tensor dz,at::Tensor wsqT,at::Tensor ab,at::Tensor dab,int64_t nsm){
 at::cuda::CUDAGuard guard(dz.device());cudaSetDevice(dz.get_device());
 int M=dz.size(0),K=dz.size(1),H=wsqT.size(0);
 TORCH_CHECK(M%128==0&&K%64==0&&H%128==0&&wsqT.size(1)==K&&ab.size(0)==M&&ab.size(1)==2*H&&dab.sizes()==ab.sizes(),"dswi shape");
 TORCH_CHECK(dz.is_contiguous()&&wsqT.is_contiguous()&&ab.is_contiguous()&&dab.is_contiguous(),"dswi layout");
 constexpr int smem=DS_ST*DS_SB+64+1024;
 cudaFuncSetAttribute(dswi,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 int grid=std::min<int>(nsm,(M/128)*(H/128));
 dswi<<<grid,384,smem,at::cuda::getCurrentCUDAStream()>>>(map(dz,64,64),map(wsqT,64,64),(bf*)ab.data_ptr(),(bf*)dab.data_ptr(),M,K,H);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void dpb_pipeline(at::Tensor d,at::Tensor v,at::Tensor p,at::Tensor dd,at::Tensor out,int64_t A,int64_t H,int64_t DH){
 at::cuda::CUDAGuard guard(d.device());cudaSetDevice(d.get_device());int L=d.size(0)/A;
 constexpr int smem=3*16384+1024;cudaFuncSetAttribute(dpb_pipe,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 dpb_pipe<<<std::min<int>(264,H*(L/64)*(L/64)),256,smem,at::cuda::getCurrentCUDAStream()>>>(map(d,64,64),map(v,64,64),(bf*)p.data_ptr(),dd.data_ptr<float>(),(bf*)out.data_ptr(),L,A,H,DH);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ __launch_bounds__(384,1) void expand_pipe(const __grid_constant__ CUtensorMap mx,const __grid_constant__ CUtensorMap mw,bf*ab,bf*h,int M,int K,int H){
 constexpr int ST=32768,NST=3;extern __shared__ __align__(1024) unsigned char sm[];
 uint64_t*full=reinterpret_cast<uint64_t*>(sm+NST*ST),*empty=full+NST;
 int tid=threadIdx.x,wg=tid/128,wtid=tid%128,warp=wtid/32,lane=tid%32,mt=M/128;
 if(!tid){for(int s=0;s<NST;++s){mbar_init(full+s,1);mbar_init(empty+s,2);}fence_barrier_init();}
 __syncthreads();unsigned it=0;
 if(!wg){
   setmaxnreg_dec<32>();
   if(!tid)for(int item=blockIdx.x;item<mt*(H/64);item+=gridDim.x){
     int m0=(item%mt)*128,n0=(item/mt)*64;
     for(int k=0;k<K;k+=64,++it){int s=it%NST;if(it>=NST)mbar_wait(empty+s,((it/NST)-1)&1);mbar_arrive_expect_tx(full+s,ST);tma_load_2d(sm+s*ST,&mx,full+s,k,m0);tma_load_2d(sm+s*ST+16384,&mw,full+s,k,n0);tma_load_2d(sm+s*ST+24576,&mw,full+s,k,H+n0);}
   }
   return;
 }
 setmaxnreg_inc<96>();int cw=wg-1;
 for(int item=blockIdx.x;item<mt*(H/64);item+=gridDim.x){
   int m0=(item%mt)*128+cw*64,n0=(item/mt)*64,prev=-1;float acc[64]={};
   for(int k=0;k<K;k+=64,++it){
     int s=it%NST;mbar_wait(full+s,(it/NST)&1);fence_regs(acc);wgmma_fence();
     #pragma unroll
     for(int ks=0;ks<4;++ks)mma_n128(acc,ds(smem_u32(sm)+s*ST+cw*8192+ks*32),ds(smem_u32(sm)+s*ST+16384+ks*32));
     wgmma_commit();if(prev>=0){wgmma_wait<1>();fence_regs(acc);if(!wtid)mbar_arrive(empty+prev);}prev=s;
   }
   wgmma_wait<0>();fence_regs(acc);if(!wtid)mbar_arrive(empty+prev);
   int r0=m0+16*warp+(lane>>2),c0=n0+2*(lane&3);
   #pragma unroll
   for(int j=0;j<8;++j)for(int rb=0;rb<2;++rb){
      int e=j*4+rb*2,r=r0+rb*8,c=c0+j*8;float a0=acc[e],a1=acc[e+1],b0=acc[e+32],b1=acc[e+33];
      *reinterpret_cast<__nv_bfloat162*>(ab+r*2*H+c)=__floats2bfloat162_rn(a0,a1);
      *reinterpret_cast<__nv_bfloat162*>(ab+r*2*H+H+c)=__floats2bfloat162_rn(b0,b1);
      *reinterpret_cast<__nv_bfloat162*>(h+r*H+c)=__floats2bfloat162_rn(a0/(1.f+__expf(-a0))*b0,a1/(1.f+__expf(-a1))*b1);
   }
 }
}
void swiglu_pipeline(at::Tensor x,at::Tensor w,at::Tensor ab,at::Tensor h){
 at::cuda::CUDAGuard guard(x.device());cudaSetDevice(x.get_device());int M=x.size(0),K=x.size(1),H=h.size(1);
 TORCH_CHECK(M%128==0&&K%64==0&&H%64==0,"SwiGLU pipeline shape");constexpr int smem=3*32768+1024;
 cudaFuncSetAttribute(expand_pipe,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 expand_pipe<<<std::min(264,(M/128)*(H/64)),384,smem,at::cuda::getCurrentCUDAStream()>>>(map(x,64,128),map(w,64,64),(bf*)ab.data_ptr(),(bf*)h.data_ptr(),M,K,H);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void pv(at::Tensor v,at::Tensor p,at::Tensor out,int64_t A,c10::optional<at::Tensor> gate,int64_t H,int64_t DH){
 at::cuda::CUDAGuard guard(v.device());cudaSetDevice(v.get_device());
 int L=v.size(0)/A; TORCH_CHECK(L%128==0&&v.scalar_type()==at::kBFloat16,"PV shape/dtype");
 auto ma=map(p,64,64),mb=map(v,64,64);int grid=std::min<int>(132,H*(L/64)*A);
 if(gate)core<false,true><<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(ma,mb,nullptr,(bf*)gate->data_ptr(),nullptr,(bf*)out.data_ptr(),L,A,H,DH,gate->stride(0),out.stride(0));
 else core<false,false><<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(ma,mb,nullptr,nullptr,nullptr,(bf*)out.data_ptr(),L,A,H,DH,0,out.stride(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void dpb(at::Tensor d,at::Tensor v,at::Tensor p,at::Tensor dd,at::Tensor out,int64_t A,int64_t H,int64_t DH){
 at::cuda::CUDAGuard guard(d.device());cudaSetDevice(d.get_device());int L=d.size(0)/A;
 auto ma=map(d,64,64),mb=map(v,64,64);int grid=std::min<int>(132,H*(L/64)*(L/64));
 core<true,false><<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(ma,mb,(bf*)p.data_ptr(),nullptr,dd.data_ptr<float>(),(bf*)out.data_ptr(),L,A,H,DH,0,0);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ __launch_bounds__(128,2) void expand(const __grid_constant__ CUtensorMap mx,const __grid_constant__ CUtensorMap mw,bf*ab,bf*h,int M,int K,int H){
 __shared__ __align__(1024) unsigned char sm[24576];__shared__ uint64_t bar;
 int tid=threadIdx.x,warp=tid/32,lane=tid%32,mt=M/64;
 if(!tid){mbar_init(&bar,1);fence_barrier_init();}__syncthreads();unsigned phase=0;
 for(int item=blockIdx.x;item<mt*(H/64);item+=gridDim.x){
   int m0=(item%mt)*64,n0=(item/mt)*64;float acc[64]={};
   for(int k=0;k<K;k+=64){
     if(!tid){mbar_arrive_expect_tx(&bar,24576);tma_load_2d(sm,&mx,&bar,k,m0);tma_load_2d(sm+8192,&mw,&bar,k,n0);tma_load_2d(sm+16384,&mw,&bar,k,H+n0);}
     mbar_wait(&bar,phase++&1);__syncthreads();fence_regs(acc);wgmma_fence();
     #pragma unroll
     for(int ks=0;ks<4;++ks)mma_n128(acc,ds(smem_u32(sm)+ks*32),ds(smem_u32(sm)+8192+ks*32));
     wgmma_commit();wgmma_wait<0>();fence_regs(acc);__syncthreads();
   }
   int r0=m0+16*warp+(lane>>2),c0=2*(lane&3);
   #pragma unroll
   for(int j=0;j<8;++j)for(int rb=0;rb<2;++rb)for(int q=0;q<2;++q){
      int e=j*4+rb*2+q,r=r0+rb*8,c=n0+j*8+c0+q;float a=acc[e],b=acc[e+32];
      ab[r*2*H+c]=__float2bfloat16_rn(a);ab[r*2*H+H+c]=__float2bfloat16_rn(b);
      h[r*H+c]=__float2bfloat16_rn(a/(1.f+__expf(-a))*b);
   } __syncthreads();
 }
}
void swiglu(at::Tensor x,at::Tensor w,at::Tensor ab,at::Tensor h){
 at::cuda::CUDAGuard guard(x.device());cudaSetDevice(x.get_device());int M=x.size(0),K=x.size(1),H=h.size(1);
 TORCH_CHECK(M%64==0&&K%64==0&&H%64==0,"SwiGLU shape");
 expand<<<std::min(132,(M/64)*(H/64)),128,0,at::cuda::getCurrentCUDAStream()>>>(map(x,64,64),map(w,64,64),(bf*)ab.data_ptr(),(bf*)h.data_ptr(),M,K,H);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("pv",&bo90::pv);m.def("grouped_pv",&bo90::grouped_pv);m.def("dpb",&bo90::dpb);m.def("swiglu",&bo90::swiglu);m.def("grouped_pv_pipeline",&bo90::grouped_pv_pipeline);m.def("pv_cta",&bo90::pv_cta);m.def("swiglu_bwd_gemm",&bo90::swiglu_bwd_gemm);m.def("dpb2",&bo90::dpb2_launch);m.def("pv_cta_tf32",&bo90::pv_cta_tf32);m.def("pv_cta_prof",&bo90::pv_cta_prof);m.def("dpb_pipeline",&bo90::dpb_pipeline);m.def("swiglu_pipeline",&bo90::swiglu_pipeline);}
#else
// pv_gate_inf.cu -- the bias-only token DiT's INFERENCE attention core, sm_100a:
//
//   a[s] = sigmoid(g[s]) * (P_h v[s]_h)      for every head h and sample s
//
// There is no query and no key: the attention weights P = softmax(pair bias) [H, L, L] (bf16, normalised) depend on neither the
// sample nor the solver step, so the runner makes them once per sample() and this kernel is a plain GEMM per head -- no online
// softmax, no running max, no rescale. v | g are column views of the v|g GEMM output [S L, 1536] bf16; a [S L, 768] bf16.
//
// Work item = (head, 128-query tile, group of SG samples); the host picks SG per shape. The item's P tile [128 i][L j] goes into
// TENSOR memory once (TMA in 64-key SW128 chunks, tcgen05.cp into the MMA's A-operand layout, L / 2 columns) and stays there while
// the group's samples stream their v tiles [128 j][48 d] through the shared-memory ring: per sample the K loop is 8 x (M 128, N 48,
// K 16) products per 128-key block, A from TMEM. Chunks are as large as a slot holds: the ring runs at a few hundred clocks per TMA
// instruction whatever its size (32-key P chunks and 64-key v tiles made it twice as slow). The loop is sample-OUTER, so one sample's accumulator is complete while the next
// sample's products run, and the epilogue of sample k (sigmoid(g) * o, g in by TMA, the gated tile out by TMA) hides under the
// MMAs of sample k + 1. (With the P tile in shared memory and the samples inner, the epilogue only started once the whole item
// was loaded and took ~40 % of the kernel.) Accumulators: two sets of 48 columns above P.
// Warps: 0 TMA producer, 1 MMA (and the P copies), 2 TMEM allocator, 3 idle, 4-7 epilogue (one query row per thread).
// -DPDL_INF (only the bf16 three-kernel inference step builds it: K2 between bo_front_bf16 and bo_tail_bf16 / bo_tail2_bf16):
// programmatic dependent launch -- launch_dependents after setup, griddepcontrol.wait before the producer's first load and before
// the epilogue reads g or stores a. Without it the cubin is the default inference / training path's, unchanged.
// -DP_EARLY (with PDL_INF; only the inference three-kernel step builds it, ``PvGateCore(p_early=True)`` from the runner): the
// producer issues the first item's P chunks BEFORE griddepcontrol.wait and waits only before its first v tile, so the P tile's
// load and its tcgen05.cp into TMEM (the MMA warp has no wait of its own) run under the front kernel instead of after it (opt2 r3:
// 2.9 of 12.5 us per CTA at L768, 1.5 of 4.1 at L384). Only P moves: v, g and the output stay after the wait. This is safe only
// because that P (the runner's hoisted P, or the per-call softmax of a hoisted bias) is written before the step and by no kernel
// of its PDL chain, and the runner launches the step's first front WITHOUT the programmatic attribute whenever its core takes this
// build, so the whole chain starts after P's writer is complete (runner.py ``_ops3b`` / ``_step3b``). The training backward's P^T
// (softmax_t right before, in the same stream) must never take it.
// L <= 768 (P plus two accumulators within 512 TMEM columns). Heads x width by -DNHEAD / -DDHEAD (16 x 48, 24 x 32, 12 x 64): a v or g
// row of the head is DH x 2 bytes inside a 128-byte swizzle row.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef SG
#define SG 2                                   // samples per work item
#endif
#ifndef NX
#define NX 3                                   // g staging tiles: loads run NX - 1 tiles ahead
#endif
#ifndef GATE
#define GATE 1                                 // 0: o = P v as it is (the training backward's dV = P^T do), no g read
#endif
#ifndef VB
#define VB 128                                 // keys per v tile (one TMA box [VB j][48 d]); L a multiple of VB
#endif
#ifndef NHEAD
#define NHEAD 16                               // heads x head width: 16 x 48, 24 x 32 or 12 x 64 (768 channels)
#endif
#ifndef DHEAD
#define DHEAD 48
#endif
constexpr int BN = VB, DH = DHEAD, QM = 128, NH = NHEAD, NQC = DH / 8;   // NQC: 16-byte chunks of a head row
static_assert(DH % 16 == 0 && DH <= 64, "head width: a multiple of 16 within one 128-byte swizzle row");
// ring slot: a v tile [VB j][48 d -> 128 B] or PCS P chunks [128 i][64 j], all SW128. Larger boxes take the SM's TMA intake from
// ~45 B/clk (12 KB) to ~60-70 (24-32 KB): with 128 keys per v tile the training shapes (A = 48) were intake-bound.
constexpr int SLOT = VB * 128 > 16384 ? VB * 128 : 16384, PCS = SLOT / 16384;
static_assert(VB % 16 == 0 && VB <= 256, "v tile");
constexpr int XG = QM * 128;                   // g in / gated o out: 16 KB
constexpr int NSLOT = (232448 - NX * XG - 1024) / SLOT;
constexpr int O_ST = 0, O_X = NSLOT * SLOT, O_BAR = O_X + NX * XG, SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_ACC = 512 - 2 * DH;       // two DH-column accumulators at the top (416 / 448); P at [0, L / 2)
constexpr uint32_t I_PV = idesc_bf16(QM, DH, 0, 1);

// o * sigmoid(g) for a pair of channels, packed f32x2: 1 / (1 + 2^(-g log2 e)) with the exponentials on MUFU and the reciprocal on
// the FMA pipe (bit-trick seed, two Newton steps: relative error <= 2e-4, below the bf16 rounding of the product). The epilogue is
// one warp per SMSP and issue-bound: per pair this is ~14 instructions where the scalar kit sigmoid (two MUFU ops per channel) was
// MUFU-bound and the scalar Newton form issue-bound at twice the count.
DEVI uint32_t gate_pair(uint32_t o0, uint32_t o1, uint32_t g2) {
#ifdef SIG_KIT
  return pack_bf16(__uint_as_float(o0) * sigmoid_kit(bf16lo(g2)), __uint_as_float(o1) * sigmoid_kit(bf16hi(g2)));
#else
  const f2 t = mul2(mk2(bf16lo(g2), bf16hi(g2)), mk2(-1.4426950408889634f, -1.4426950408889634f));
  const f2 e = add2(mk2(ex2f(lo2(t)), ex2f(hi2(t))), mk2(1.f, 1.f));
  const f2 d = mk2(fmin_nan(lo2(e), RCP_SEED_MAX), fmin_nan(hi2(e), RCP_SEED_MAX));   // g <= -87.3: keep the seed valid
  const f2 nd = neg2(d), two = mk2(2.f, 2.f);
  f2 r = mk2(__int_as_float(0x7EF311C3 - __float_as_int(lo2(d))), __int_as_float(0x7EF311C3 - __float_as_int(hi2(d))));
  r = mul2(r, fma2(nd, r, two));
  r = mul2(r, fma2(nd, r, two));
  const f2 o = mul2(mk2u(o0, o1), r);
  return pack_bf16(lo2(o), hi2(o));
#endif
}
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }

#ifdef TRACE
// -DTRACE: %globaltimer (ns) into g_trace[16][2048] for the 16 virtual CTAs [TRACE_CTA0, + 16) (bo_tail_tf32.cu's layout, read by
// tf32.read_trace). Per CTA: 0 start (after setup), 1 producer past the PDL wait, 2 MMA warp done, 3 epilogue done (stores read);
// per ring position g (P chunks and v tiles in one sequence): EV_PW producer begins the empty wait, EV_PI producer issued the TMA,
// EV_MW MMA begins the full wait, EV_MS MMA saw it full, EV_MI MMA issued its products / copies; per (item, sample) tile t: EV_AW
// MMA begins the accumulator-empty wait, EV_AS MMA saw it empty, EV_EA epilogue saw the accumulator full, EV_EG epilogue saw g,
// EV_ES store issued; per item li: EV_FW MMA begins the p_free wait, EV_FS MMA saw it.
__device__ unsigned long long g_trace[16 * 2048];
#ifndef TRACE_CTA0
#define TRACE_CTA0 0
#endif
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
constexpr int EV_PW = 256, EV_PI = 512, EV_MW = 768, EV_MS = 1024, EV_MI = 1280, EV_AW = 1536, EV_AS = 1600, EV_EA = 1664,
              EV_EG = 1728, EV_ES = 1792, EV_FW = 1856, EV_FS = 1920;
#define TR(ev, i, lim) do { const int tc_ = vb - TRACE_CTA0; if (tc_ >= 0 && tc_ < 16 && (i) < (lim)) g_trace[tc_ * 2048 + (ev) + (i)] = gtime(); } while (0)
#else
#define TR(ev, i, lim) do { } while (0)
#endif

struct Bars {
  uint64_t full[NSLOT], empty[NSLOT], acc_full[2], acc_empty[2], g_full[NX], p_free;
  uint32_t tmem;
};

// The kernel's body on a virtual grid (CTA vb of vg): bo_pv_gate_inf_sm100 below runs it on the real one, bo_pvdpb.cu next to the
// bias gradient's in one launch (-DBODY_ONLY leaves the kernel out). The maps are the kernel's __grid_constant__ parameters.
DEVI void pv_gate_body(const CUtensorMap& mp, const CUtensorMap& mv, const CUtensorMap& mg, const CUtensorMap& mo, int L, int S,
                       const int vb, const int vg) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = L / QM, nb = L / BN, nc = L / 64, ng = (S + SG - 1) / SG;
  const int items = NH * mt * ng;
  const int my = (items > vb) ? (items - vb + vg - 1) / vg : 0;
  // item li of this CTA -> first sample a0, samples ns, query tile m0, head h (the group fastest)
  auto item_of = [&](int li, int& a0, int& ns, int& m0, int& h) {
    const int wi = vb + li * vg;
    a0 = (wi % ng) * SG; ns = min(SG, S - a0);
    const int r = wi / ng;
    m0 = (r % mt) * QM; h = r / mt;
  };

  if (tid == 0) {
    for (int s = 0; s < NSLOT; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    for (int x = 0; x < NX; ++x) mbar_init(&B.g_full[x], 1);
    mbar_init(&B.p_free, 1);
    fence_barrier_init();
    prefetch_map(&mp); prefetch_map(&mv); prefetch_map(&mg); prefetch_map(&mo);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (tid == 0) TR(0, 0, 1);
#ifdef PDL_INF
  pdl_launch();
#endif

  if (warp == 0) {
    if (lane == 0) {
#if defined(PDL_INF) && !defined(P_EARLY)
      pdl_wait();                                                 // v: the v|g GEMM's output (P: hoisted, but in the chain's order)
#endif
      TR(1, 0, 1);
      int g = 0;
      auto slot = [&](uint32_t bytes) {                           // the next ring slot, empty, armed for `bytes`
        const int s = g % NSLOT;
        TR(EV_PW, g, 256);
        if (g >= NSLOT) mbar_wait(&B.empty[s], ((g / NSLOT) - 1) & 1);
        mbar_expect_tx(&B.full[s], bytes);
        ++g;
        return s;
      };
      for (int li = 0; li < my; ++li) {
        int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
        for (int c = 0; c < nc; c += PCS) {                      // the P tile, 64 PCS keys per slot
          const int s = slot(PCS * QM * 64 * 2);
          for (int e = 0; e < PCS; ++e) tma_load_2d(su + O_ST + s * SLOT + e * 16384, &mp, &B.full[s], (c + e) * 64, h * L + m0);
          TR(EV_PI, g - 1, 256);
        }
#if defined(PDL_INF) && defined(P_EARLY)
        if (li == 0) pdl_wait();                                  // v: the v|g GEMM's output; P (hoisted, read-only) went first
#endif
        for (int k = 0; k < ns; ++k)
          for (int n = 0; n < nb; ++n) {
            const int s = slot(BN * DH * 2);
            tma_load_2d(su + O_ST + s * SLOT, &mv, &B.full[s], h * DH, (a0 + k) * L + n * BN);
            TR(EV_PI, g - 1, 256);
          }
      }
    }
  } else if (warp == 1) {
    int g = 0, t = 0;
    for (int li = 0; li < my; ++li) {
      int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
      if (lane == 0) TR(EV_FW, li, 64);
      if (li >= 1) mbar_wait(&B.p_free, (li - 1) & 1);            // the previous item's products have read P
      if (lane == 0) TR(EV_FS, li, 64);
      for (int c = 0; c < nc; c += PCS, ++g) {                    // P chunk c -> TMEM columns 32 c .. 32 c + 31
        const int s = g % NSLOT;
        if (lane == 0) TR(EV_MW, g, 256);
        mbar_wait(&B.full[s], (g / NSLOT) & 1);
        if (lane == 0) TR(EV_MS, g, 256);
        tc_fence_after();
        if (elect_one()) {
          for (int e = 0; e < PCS; ++e) {
            const uint64_t dp = desc_k128(su + O_ST + s * SLOT + e * 16384);
#pragma unroll
            for (int j = 0; j < 4; ++j) tmem_cp_128x256b(tmem + (c + e) * 32 + j * 8, dp + (uint64_t)(j * 2));   // 16 keys per copy
          }
          tc_commit(&B.empty[s]);
          TR(EV_MI, g, 256);
        }
        __syncwarp();
      }
      for (int k = 0; k < ns; ++k, ++t) {
        const int b = t & 1;
        if (lane == 0) TR(EV_AW, t, 64);
        if (t >= 2) mbar_wait(&B.acc_empty[b], ((t >> 1) - 1) & 1);
        if (lane == 0) TR(EV_AS, t, 64);
        tc_fence_after();
        for (int n = 0; n < nb; ++n, ++g) {
          const int s = g % NSLOT;
          if (lane == 0) TR(EV_MW, g, 256);
          mbar_wait(&B.full[s], (g / NSLOT) & 1);
          if (lane == 0) TR(EV_MS, g, 256);
          tc_fence_after();
          const uint64_t dv = desc_mn128(su + O_ST + s * SLOT, 16384);
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < BN / 16; ++ks)
              umma_ts(tmem + T_ACC + b * DH, tmem + n * (BN / 2) + ks * 8, dv + (uint64_t)(ks * 2048 >> 4), I_PV, (n > 0 || ks > 0) ? 1u : 0u);
            tc_commit(&B.empty[s]);
            if (n == nb - 1) {
              tc_commit(&B.acc_full[b]);
              if (k == ns - 1) tc_commit(&B.p_free);
            }
            TR(EV_MI, g, 256);
          }
          __syncwarp();
        }
      }
    }
    if (lane == 0) TR(2, 0, 1);
  } else if (warp >= 4) {
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    // NX g staging tiles rotate over the running sequence of (item, sample) tiles. Thread r == 0 issues the store of tile t, then
    // waits for the store of tile t - 1 (not the one just issued) to have read its buffer and loads tile t - 1 + NX's g there.
#ifdef PDL_INF
    pdl_wait();                                                   // g: the v|g GEMM's output; a: may still be read by an earlier kernel
#endif
    int pli = 0, pk = 0;                                          // the next tile whose g is loaded
    auto load_next_g = [&](int buf) {                             // thread r == 0 only
      if (pli >= my) return;
#if GATE
      int a0, ns, m0, h; item_of(pli, a0, ns, m0, h);
      mbar_expect_tx(&B.g_full[buf], QM * DH * 2);
      tma_load_2d(su + O_X + buf * XG, &mg, &B.g_full[buf], h * DH, (a0 + pk) * L + m0);
#else
      int a0, ns, m0, h; item_of(pli, a0, ns, m0, h);
      mbar_arrive(&B.g_full[buf]);                                // the staging tile is only an output: nothing to load
#endif
      if (++pk == ns) { pk = 0; ++pli; }
    };
    if (r == 0) for (int x = 0; x < NX - 1; ++x) load_next_g(x);
    int t = 0;
    for (int li = 0; li < my; ++li) {
      int a0, ns, m0, h; item_of(li, a0, ns, m0, h);
      for (int k = 0; k < ns; ++k, ++t) {
        const int b = t & 1, xb = t % NX;
        const uint32_t xg = su + O_X + xb * XG;
        mbar_wait(&B.acc_full[b], (t >> 1) & 1);
        if (r == 0) TR(EV_EA, t, 64);
        tc_fence_after();
        uint32_t ov[DH];
#pragma unroll
        for (int cc = 0; cc < DH / 16; ++cc) tmem_ld16(trow + T_ACC + b * DH + cc * 16, *reinterpret_cast<uint32_t(*)[16]>(ov + 16 * cc));
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[b]);
        mbar_wait(&B.g_full[xb], (t / NX) & 1);
        if (r == 0) TR(EV_EG, t, 64);
        // all six loads first, then the math, then the stores: the shared accesses are volatile asm with a memory clobber, so an
        // interleaved load / compute / store per chunk serialises six load latencies
        uint4 gw[NQC];
#if GATE
#pragma unroll
        for (int q = 0; q < NQC; ++q) gw[q] = lds128(xg + sw128(r, q));   // 8 channels per 16-byte chunk of this thread's row
#else
#pragma unroll
        for (int q = 0; q < NQC; ++q)
          gw[q] = make_uint4(pack_bf16(__uint_as_float(ov[8 * q]), __uint_as_float(ov[8 * q + 1])),
                             pack_bf16(__uint_as_float(ov[8 * q + 2]), __uint_as_float(ov[8 * q + 3])),
                             pack_bf16(__uint_as_float(ov[8 * q + 4]), __uint_as_float(ov[8 * q + 5])),
                             pack_bf16(__uint_as_float(ov[8 * q + 6]), __uint_as_float(ov[8 * q + 7])));
#endif
#pragma unroll
        for (int q = 0; q < (GATE ? NQC : 0); ++q) {
          const uint32_t gg[4] = {gw[q].x, gw[q].y, gw[q].z, gw[q].w};
          uint32_t o4[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) o4[e] = gate_pair(ov[8 * q + 2 * e], ov[8 * q + 2 * e + 1], gg[e]);
          gw[q] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
#pragma unroll
        for (int q = 0; q < NQC; ++q) sts128(xg + sw128(r, q), gw[q]);
        fence_proxy_async();
        named_bar_sync(1, 128);
        if (r == 0) {
          tma_store_2d(&mo, xg, h * DH, (a0 + k) * L + m0);
          tma_store_commit();
          TR(EV_ES, t, 64);
          if (t == 0) load_next_g(NX - 1);                        // its buffer has never held a tile
          else { tma_store_wait_read1(); load_next_g((t - 1) % NX); }
        }
      }
    }
    if (r == 0) { tma_store_wait0(); TR(3, 0, 1); }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

#ifndef BODY_ONLY
extern "C" __global__ void __launch_bounds__(256, 1)
bo_pv_gate_inf_sm100(const __grid_constant__ CUtensorMap mp, const __grid_constant__ CUtensorMap mv,
                     const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mo, int L, int S) {
  pv_gate_body(mp, mv, mg, mo, L, S, (int)blockIdx.x, (int)gridDim.x);
}
#endif
#endif // MINIWORLD_BO_SM90
