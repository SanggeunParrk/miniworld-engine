// E113: reduce PWA parameter-gradient slabs directly into leaf dtypes.
// Association exactly matches reduce_rows (8 row groups, then 0..7).
// Pair products exactly match pwa_finalize_typed_kernel, including F32
// multiply/add rounding and h%4 head accumulation. No persistent outputs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
namespace pwa_grad_reduce80 {
struct Outputs {void* data[8];unsigned bf16;};
__device__ __forceinline__ void store(Outputs out,int p,int64_t i,float v) {
 if(out.bf16&(1u<<p))reinterpret_cast<__nv_bfloat16*>(out.data[p])[i]=__float2bfloat16_rn(v);
 else reinterpret_cast<float*>(out.data[p])[i]=v;
}
__global__ void __launch_bounds__(256) fused_kernel(
 const float* glue,const float* dgv,const float* pm,const float* ps,
 const float* gamma,const float* beta,const __nv_bfloat16* wb,
 Outputs out,int rg,int rd,int rp,int dm,int hc,int dz) {
 __shared__ float shared[8][8][32];
 __shared__ float scalar[8][8][32];
 const int lane=threadIdx.x,rowgroup=threadIdx.y;
 const int gch=hc<64?hc:64,gw=2*dm*hc,dw=dm*hc+2*dm;
 const int gb=(gw+127)/128,db=(dw+127)/128;
 if(blockIdx.x<(unsigned)(gb+db)) {
  const bool isglue=blockIdx.x<(unsigned)gb;
  const int w=isglue?gw:dw,r=isglue?rg:rd;
  const float* src=isglue?glue:dgv;
  const int col=(isglue?blockIdx.x:blockIdx.x-gb)*128+lane*4;
  float4 sum=make_float4(0.f,0.f,0.f,0.f);
  if(col<w)for(int i=rowgroup;i<r;i+=8) {
   const float4 v=*reinterpret_cast<const float4*>(src+(long)i*w+col);
   sum.x=__fadd_rn(sum.x,v.x);sum.y=__fadd_rn(sum.y,v.y);
   sum.z=__fadd_rn(sum.z,v.z);sum.w=__fadd_rn(sum.w,v.w);
  }
  shared[rowgroup][0][lane]=sum.x;shared[rowgroup][1][lane]=sum.y;
  shared[rowgroup][2][lane]=sum.z;shared[rowgroup][3][lane]=sum.w;
  __syncthreads();
  if(rowgroup==0 && col<w) {
#pragma unroll
   for(int c=0;c<4;++c) {
    float value=shared[0][c][lane];
#pragma unroll
    for(int g=1;g<8;++g)value=__fadd_rn(value,shared[g][c][lane]);
    const int index=col+c;
    if(isglue) {
     const int group=index/(2*dm*gch),inner=index%(2*dm*gch);
     if(inner<dm*gch)store(out,7,(inner/gch)*hc+group*gch+inner%gch,value);
     else store(out,3,group*dm*gch+inner-dm*gch,value);
    } else {
     if(index<dm*hc)store(out,2,index,value);
     else if(index<dm*hc+dm)store(out,0,index-dm*hc,value);
     else store(out,1,index-dm*hc-dm,value);
    }
   }
  }
 } else {
  const int col=(blockIdx.x-gb-db)*32+lane;
  float m[8],s[8];
#pragma unroll
  for(int h=0;h<8;++h){m[h]=0.f;s[h]=0.f;}
  if(col<dz)for(int r=rowgroup;r<rp;r+=8) {
#pragma unroll
   for(int h=0;h<8;++h) {
    m[h]=__fadd_rn(m[h],pm[((long)r*8+h)*dz+col]);
    s[h]=__fadd_rn(s[h],ps[(long)r*8+h]);
   }
  }
#pragma unroll
  for(int h=0;h<8;++h){shared[rowgroup][h][lane]=m[h];scalar[rowgroup][h][lane]=s[h];}
  __syncthreads();
  if(rowgroup==0 && col<dz) {
   float am[4]={0.f,0.f,0.f,0.f},as[4]={0.f,0.f,0.f,0.f};
#pragma unroll
   for(int h=0;h<8;++h) {
    float mv=shared[0][h][lane],sv=scalar[0][h][lane];
#pragma unroll
    for(int g=1;g<8;++g){mv=__fadd_rn(mv,shared[g][h][lane]);sv=__fadd_rn(sv,scalar[g][h][lane]);}
    const float weight=__bfloat162float(wb[h*dz+col]);
    am[h%4]=__fadd_rn(am[h%4],__fmul_rn(weight,mv));
    as[h%4]=__fadd_rn(as[h%4],__fmul_rn(weight,sv));
    store(out,6,h*dz+col,__fadd_rn(__fmul_rn(gamma[col],mv),__fmul_rn(beta[col],sv)));
   }
   store(out,4,col,__fadd_rn(__fadd_rn(__fadd_rn(am[0],am[1]),am[2]),am[3]));
   store(out,5,col,__fadd_rn(__fadd_rn(__fadd_rn(as[0],as[1]),as[2]),as[3]));
  }
 }
}
} // namespace pwa_grad_reduce80
std::vector<torch::Tensor> result(torch::Tensor glue,torch::Tensor dgv,torch::Tensor pm,torch::Tensor ps,torch::Tensor gamma,torch::Tensor beta,torch::Tensor wb,int64_t dm,int64_t hc,int64_t dz,int64_t mask) {
 const at::cuda::CUDAGuard guard(glue.device());
 for(auto t:{glue,dgv,pm,ps,gamma,beta})TORCH_CHECK(t.is_cuda() && t.device()==glue.device() && t.is_contiguous() && t.scalar_type()==at::kFloat,"PWA fused gradient F32 buffers");
 TORCH_CHECK(wb.is_cuda() && wb.device()==glue.device() && wb.is_contiguous() && wb.scalar_type()==at::kBFloat16,"PWA fused gradient BF16 Wb");
 TORCH_CHECK(dm==64 && hc==256 && dz==128 && mask>=0 && mask<256,"PWA fused gradient frozen widths/dtype bits");
 TORCH_CHECK(glue.dim()==2 && glue.size(1)==2*dm*hc && dgv.dim()==2 && dgv.size(1)==dm*hc+2*dm && pm.dim()==2 && pm.size(1)==8*dz && ps.dim()==2 && ps.size(0)==pm.size(0) && ps.size(1)==8 && gamma.numel()==dz && beta.numel()==dz && wb.sizes()==torch::IntArrayRef({8,dz}),"PWA fused gradient physical shapes");
 TORCH_CHECK(glue.size(0)>0 && dgv.size(0)>0 && pm.size(0)>0,"PWA fused gradient nonempty reductions");
 std::vector<torch::Tensor> out;
 const std::vector<std::vector<int64_t>> sizes={{dm},{dm},{hc,dm},{hc,dm},{dz},{dz},{8,dz},{dm,hc}};
 pwa_grad_reduce80::Outputs p;p.bf16=(unsigned)mask;
 for(int i=0;i<8;++i){out.push_back(torch::empty(sizes[i],glue.options().dtype((mask&(1<<i))?at::kBFloat16:at::kFloat)));p.data[i]=out.back().data_ptr();}
 const int gb=(2*dm*hc+127)/128,db=(dm*hc+2*dm+127)/128,pb=(dz+31)/32;
 pwa_grad_reduce80::fused_kernel<<<gb+db+pb,dim3(32,8),0,at::cuda::getCurrentCUDAStream()>>>(glue.data_ptr<float>(),dgv.data_ptr<float>(),pm.data_ptr<float>(),ps.data_ptr<float>(),gamma.data_ptr<float>(),beta.data_ptr<float>(),(const __nv_bfloat16*)wb.data_ptr(),p,glue.size(0),dgv.size(0),pm.size(0),dm,hc,dz);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("result",&result);}
