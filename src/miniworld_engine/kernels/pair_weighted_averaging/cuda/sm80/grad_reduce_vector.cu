// E115: reduce PWA parameter-gradient slabs directly into leaf dtypes.
// Association exactly matches reduce_rows (8 row groups, then 0..7).
// Pair products exactly match pwa_finalize_typed_kernel, including F32
// multiply/add rounding and h%4 head accumulation. No persistent outputs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
namespace pwa_grad_reduce_vector80 {
struct Outputs {void* data[8];unsigned bf16;};
__device__ __forceinline__ void store(Outputs out,int p,int64_t i,float v) {
 if(out.bf16&(1u<<p))reinterpret_cast<__nv_bfloat16*>(out.data[p])[i]=__float2bfloat16_rn(v);
 else reinterpret_cast<float*>(out.data[p])[i]=v;
}
__device__ __forceinline__ void store4(Outputs out,int p,int64_t i,float4 v) {
 if(out.bf16&(1u<<p)) {
  const __nv_bfloat162 a=__floats2bfloat162_rn(v.x,v.y),b=__floats2bfloat162_rn(v.z,v.w);
  *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(out.data[p])+i)=make_uint2(*reinterpret_cast<const unsigned*>(&a),*reinterpret_cast<const unsigned*>(&b));
 } else *reinterpret_cast<float4*>(reinterpret_cast<float*>(out.data[p])+i)=v;
}
__global__ void __launch_bounds__(256) fused_kernel(
 const float* glue,const float* dgv,const float* pm,const float* ps,
 const float* gamma,const float* beta,const __nv_bfloat16* wb,
 Outputs out,int rg,int rd,int rp,int dm,int hc,int dz) {
 __shared__ float shared[8][128];
 __shared__ float scalar[8][8];
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
  shared[rowgroup][lane]=sum.x;shared[rowgroup][32+lane]=sum.y;
  shared[rowgroup][64+lane]=sum.z;shared[rowgroup][96+lane]=sum.w;
  __syncthreads();
  if(rowgroup==0 && col<w) {
   float4 value=make_float4(shared[0][lane],shared[0][32+lane],shared[0][64+lane],shared[0][96+lane]);
#pragma unroll
   for(int g=1;g<8;++g) {
    value.x=__fadd_rn(value.x,shared[g][lane]);value.y=__fadd_rn(value.y,shared[g][32+lane]);
    value.z=__fadd_rn(value.z,shared[g][64+lane]);value.w=__fadd_rn(value.w,shared[g][96+lane]);
   }
   if(isglue) {
    const int group=col/(2*dm*gch),inner=col%(2*dm*gch);
    if(inner<dm*gch)store4(out,7,(inner/gch)*hc+group*gch+inner%gch,value);
    else store4(out,3,group*dm*gch+inner-dm*gch,value);
   } else {
    if(col<dm*hc)store4(out,2,col,value);
    else if(col<dm*hc+dm)store4(out,0,col-dm*hc,value);
    else store4(out,1,col-dm*hc-dm,value);
   }
  }
 } else {
  // Eight heads x four column quads per warp: each thread reduces only
  // one float4. Eight CTAs cover all128 columns, twice the first candidate.
  const int head=lane/4,quad=lane%4;
  const int col=(blockIdx.x-gb-db)*16+quad*4;
  float4 m=make_float4(0.f,0.f,0.f,0.f);float sum=0.f;
  if(col<dz)for(int r=rowgroup;r<rp;r+=8) {
   const float4 v=*reinterpret_cast<const float4*>(pm+((long)r*8+head)*dz+col);
   m.x=__fadd_rn(m.x,v.x);m.y=__fadd_rn(m.y,v.y);m.z=__fadd_rn(m.z,v.z);m.w=__fadd_rn(m.w,v.w);
   if(quad==0)sum=__fadd_rn(sum,ps[(long)r*8+head]);
  }
  const int offset=head*16+quad*4;
  shared[rowgroup][offset]=m.x;shared[rowgroup][offset+1]=m.y;
  shared[rowgroup][offset+2]=m.z;shared[rowgroup][offset+3]=m.w;
  if(quad==0)scalar[rowgroup][head]=sum;
  __syncthreads();
  if(rowgroup==0 && col<dz) {
   float mv[4],sv=scalar[0][head];
#pragma unroll
   for(int c=0;c<4;++c)mv[c]=shared[0][offset+c];
#pragma unroll
   for(int g=1;g<8;++g) {
    sv=__fadd_rn(sv,scalar[g][head]);
#pragma unroll
    for(int c=0;c<4;++c)mv[c]=__fadd_rn(mv[c],shared[g][offset+c]);
   }
   float wvalue[4],bvalue[4],products_m[4],products_s[4];
#pragma unroll
   for(int c=0;c<4;++c) {
    const float weight=__bfloat162float(wb[head*dz+col+c]);
    products_m[c]=__fmul_rn(weight,mv[c]);products_s[c]=__fmul_rn(weight,sv);
    bvalue[c]=__fadd_rn(__fmul_rn(gamma[col+c],mv[c]),__fmul_rn(beta[col+c],sv));
   }
   store4(out,6,head*dz+col,make_float4(bvalue[0],bvalue[1],bvalue[2],bvalue[3]));
   float am[4][4],as[4][4];
#pragma unroll
   for(int h=0;h<4;++h)
#pragma unroll
    for(int c=0;c<4;++c){am[h][c]=0.f;as[h][c]=0.f;}
#pragma unroll
   for(int h=0;h<8;++h)
#pragma unroll
    for(int c=0;c<4;++c) {
     const float p=__shfl_sync(0xffffffffu,products_m[c],h*4+quad);
     const float q=__shfl_sync(0xffffffffu,products_s[c],h*4+quad);
     am[h%4][c]=__fadd_rn(am[h%4][c],p);as[h%4][c]=__fadd_rn(as[h%4][c],q);
    }
   if(head==0) {
    float mval[4],sval[4];
#pragma unroll
    for(int c=0;c<4;++c) {
     mval[c]=__fadd_rn(__fadd_rn(__fadd_rn(am[0][c],am[1][c]),am[2][c]),am[3][c]);
     sval[c]=__fadd_rn(__fadd_rn(__fadd_rn(as[0][c],as[1][c]),as[2][c]),as[3][c]);
    }
    store4(out,4,col,make_float4(mval[0],mval[1],mval[2],mval[3]));
    store4(out,5,col,make_float4(sval[0],sval[1],sval[2],sval[3]));
   }
  }
 }
}
} // namespace pwa_grad_reduce_vector80
std::vector<torch::Tensor> result(torch::Tensor glue,torch::Tensor dgv,torch::Tensor pm,torch::Tensor ps,torch::Tensor gamma,torch::Tensor beta,torch::Tensor wb,int64_t dm,int64_t hc,int64_t dz,int64_t mask) {
 const at::cuda::CUDAGuard guard(glue.device());
 for(auto t:{glue,dgv,pm,ps,gamma,beta})TORCH_CHECK(t.is_cuda() && t.device()==glue.device() && t.is_contiguous() && t.scalar_type()==at::kFloat,"PWA fused gradient F32 buffers");
 TORCH_CHECK(wb.is_cuda() && wb.device()==glue.device() && wb.is_contiguous() && wb.scalar_type()==at::kBFloat16,"PWA fused gradient BF16 Wb");
 TORCH_CHECK(dm==64 && hc==256 && dz==128 && mask>=0 && mask<256,"PWA fused gradient frozen widths/dtype bits");
 TORCH_CHECK(glue.dim()==2 && glue.size(1)==2*dm*hc && dgv.dim()==2 && dgv.size(1)==dm*hc+2*dm && pm.dim()==2 && pm.size(1)==8*dz && ps.dim()==2 && ps.size(0)==pm.size(0) && ps.size(1)==8 && gamma.numel()==dz && beta.numel()==dz && wb.sizes()==torch::IntArrayRef({8,dz}),"PWA fused gradient physical shapes");
 TORCH_CHECK(glue.size(0)>0 && dgv.size(0)>0 && pm.size(0)>0,"PWA fused gradient nonempty reductions");
 std::vector<torch::Tensor> out;
 const std::vector<std::vector<int64_t>> sizes={{dm},{dm},{hc,dm},{hc,dm},{dz},{dz},{8,dz},{dm,hc}};
 pwa_grad_reduce_vector80::Outputs p;p.bf16=(unsigned)mask;
 for(int i=0;i<8;++i){out.push_back(torch::empty(sizes[i],glue.options().dtype((mask&(1<<i))?at::kBFloat16:at::kFloat)));p.data[i]=out.back().data_ptr();}
 const int gb=(2*dm*hc+127)/128,db=(dm*hc+2*dm+127)/128,pb=(dz+15)/16;
 pwa_grad_reduce_vector80::fused_kernel<<<gb+db+pb,dim3(32,8),0,at::cuda::getCurrentCUDAStream()>>>(glue.data_ptr<float>(),dgv.data_ptr<float>(),pm.data_ptr<float>(),ps.data_ptr<float>(),gamma.data_ptr<float>(),beta.data_ptr<float>(),(const __nv_bfloat16*)wb.data_ptr(),p,glue.size(0),dgv.size(0),pm.size(0),dm,hc,dz);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("result",&result);}
