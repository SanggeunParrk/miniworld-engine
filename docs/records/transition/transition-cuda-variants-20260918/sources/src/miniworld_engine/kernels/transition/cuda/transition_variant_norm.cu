// Shared LN for both Transition variants: FP32 affine and fused identity dx.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
using BF=__nv_bfloat16;
__device__ float sum_warp(float x){
    #pragma unroll
    for(int delta=16;delta;delta/=2)x+=__shfl_xor_sync(0xffffffff,x,delta);
    return x;
}
template<int D> __global__ void norm_forward(const BF* x,const float* w,const float* b,BF* y,float* mean,float* rstd,int M,float eps){
    int lane=threadIdx.x%32,row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(row>=M)return;
    constexpr int N=D/32;float v[N],s=0;
    #pragma unroll
    for(int j=0;j<N;++j){v[j]=__bfloat162float(x[row*D+lane+32*j]);s+=v[j];}
    float mu=sum_warp(s)/D,var=0;
    #pragma unroll
    for(int j=0;j<N;++j){float z=v[j]-mu;var+=z*z;}
    float rs=rsqrtf(sum_warp(var)/D+eps);
    if(lane==0){mean[row]=mu;rstd[row]=rs;}
    #pragma unroll
    for(int j=0;j<N;++j){int c=lane+32*j;y[row*D+c]=__float2bfloat16((v[j]-mu)*rs*w[c]+b[c]);}
}
template<int D,int TX> __global__ void norm_backward(const BF* dy,const BF* x,const float* w,const float* mean,const float* rstd,const BF* residual,BF* dx,float* pg,float* pb,int M){
    constexpr int E=TX/2,NV=(D+32*E-1)/(32*E),N=NV*E;
    using V=typename std::conditional<TX==16,uint4,uint2>::type;
    union Pack{V v;BF a[E];};
    int lane=threadIdx.x%32,warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,nwarps=gridDim.x*(blockDim.x/32);
    float ag[N]={},ab[N]={},gamma[N]={};
    #pragma unroll
    for(int v=0;v<NV;++v){int c=(v*32+lane)*E;
        if(c<D){
            #pragma unroll
            for(int e=0;e<E;++e)gamma[v*E+e]=w[c+e];
        }
    }
    for(int row=warp;row<M;row+=nwarps){
        float xv[N],gv[N],wv[N],s=0,sx=0,rs=rstd[row],mu=mean[row];
        #pragma unroll
        for(int v=0;v<NV;++v){int c=(v*32+lane)*E;
            if(c<D){Pack xp,dp;xp.v=*reinterpret_cast<const V*>(x+row*D+c);dp.v=*reinterpret_cast<const V*>(dy+row*D+c);
                #pragma unroll
                for(int e=0;e<E;++e){int j=v*E+e;xv[j]=(__bfloat162float(xp.a[e])-mu)*rs;gv[j]=__bfloat162float(dp.a[e]);wv[j]=gv[j]*gamma[j];s+=wv[j];sx+=wv[j]*xv[j];}
            }
        }
        s=sum_warp(s)/D;sx=sum_warp(sx)/D;
        #pragma unroll
        for(int v=0;v<NV;++v){int c=(v*32+lane)*E;
            if(c<D){Pack op,rp;rp.v=*reinterpret_cast<const V*>(residual+row*D+c);
                #pragma unroll
                for(int e=0;e<E;++e){int j=v*E+e;BF rounded=__float2bfloat16((wv[j]-s-xv[j]*sx)*rs);op.a[e]=__float2bfloat16(__bfloat162float(rounded)+__bfloat162float(rp.a[e]));ag[j]+=gv[j]*xv[j];ab[j]+=gv[j];}
                *reinterpret_cast<V*>(dx+row*D+c)=op.v;
            }
        }
    }
    #pragma unroll
    for(int v=0;v<NV;++v){int c=(v*32+lane)*E;
        if(c<D){
            #pragma unroll
            for(int e=0;e<E;++e){pg[warp*D+c+e]=ag[v*E+e];pb[warp*D+c+e]=ab[v*E+e];}
        }
    }
}
__global__ void norm_reduce(const float* pg,const float* pb,float* dg,float* db,int rows,int d,int channels){
    // Trade channel coalescing for more independent CTAs at small D. A CTA
    // owns `channels` adjacent columns; remaining threads split partial rows.
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int c=blockIdx.x*channels+threadIdx.x%channels;
    int row0=threadIdx.x/channels,step=blockDim.x/channels;
    float g=0,b=0;
    if(c<d)for(int r=row0;r<rows;r+=step){g+=pg[r*d+c];b+=pb[r*d+c];}
    for(int delta=16;delta>=channels;delta/=2){
        g+=__shfl_down_sync(0xffffffff,g,delta);
        b+=__shfl_down_sync(0xffffffff,b,delta);
    }
    extern __shared__ float sm[];
    int count=(blockDim.x/32)*channels;
    if(lane<channels){sm[warp*channels+lane]=g;sm[count+warp*channels+lane]=b;}
    __syncthreads();
    if(threadIdx.x<channels && c<d){
        g=0;b=0;
        for(int w=0;w<blockDim.x/32;++w){g+=sm[w*channels+threadIdx.x];b+=sm[count+w*channels+threadIdx.x];}
        dg[c]=g;db[c]=b;
    }
}
// Resolve the registered CUDA kernel before its first launch. In the tested
// PyTorch/runtime combination, launching an unresolved native LN kernel caused
// an internal cuKernelGetFunction invalid-handle report under memcheck.
// Keep each function's attribute constant across configs and graph replays.
template<class Kernel> void initialize_kernel(Kernel kernel,int shared_bytes=0){
    auto err=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes);
    TORCH_CHECK(err==cudaSuccess,"LN kernel initialization: ",cudaGetErrorString(err));
}
void check_norm(torch::Tensor x,torch::Tensor w){
    TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.is_contiguous() && x.scalar_type()==at::kBFloat16,"x must be CUDA contiguous BF16 [M,D]");
    TORCH_CHECK(w.device()==x.device() && w.is_contiguous() && w.dim()==1 && w.size(0)==x.size(1) && w.scalar_type()==at::kFloat,"affine must be FP32 [D]");
    TORCH_CHECK(x.size(1)==128 || x.size(1)==256 || x.size(1)==384 || x.size(1)==512,"unsupported D");
}
std::vector<torch::Tensor> forward(torch::Tensor x,torch::Tensor w,torch::Tensor b,double eps,int warps){
    check_norm(x,w);check_norm(x,b);TORCH_CHECK(warps==4 || warps==8,"warps must be 4/8");
    c10::cuda::CUDAGuard guard(x.device());int m=x.size(0),d=x.size(1);auto y=torch::empty_like(x);auto mean=torch::empty({m},w.options()),rs=torch::empty_like(mean);
    if(m){auto stream=at::cuda::getCurrentCUDAStream();
        #define FWD(D) case D: initialize_kernel(norm_forward<D>); norm_forward<D><<<(m+warps-1)/warps,32*warps,0,stream>>>((BF*)x.data_ptr(),w.data_ptr<float>(),b.data_ptr<float>(),(BF*)y.data_ptr(),mean.data_ptr<float>(),rs.data_ptr<float>(),m,eps);break;
        switch(d){FWD(128) FWD(256) FWD(384) FWD(512)}
        #undef FWD
        TORCH_CHECK(cudaGetLastError()==cudaSuccess,"LN forward launch failed");
    }return {y,mean,rs};
}
std::vector<torch::Tensor> backward(torch::Tensor dy,torch::Tensor x,torch::Tensor w,torch::Tensor mean,torch::Tensor rs,torch::Tensor residual,int warps,int waves,int tx,int reduce_threads,int reduce_channels){
    check_norm(x,w);
    for(auto t:{dy,residual})TORCH_CHECK(t.sizes()==x.sizes() && t.device()==x.device() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),"LN gradients must match x");
    for(auto t:{mean,rs})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==at::kFloat && t.is_contiguous() && t.dim()==1 && t.size(0)==x.size(0),"LN statistics must be FP32 [M]");
    TORCH_CHECK((warps==4 || warps==8) && (waves==2 || waves==4 || waves==8) && (tx==8 || tx==16) && (reduce_threads==128 || reduce_threads==256) && (reduce_channels==1 || reduce_channels==4 || reduce_channels==16 || reduce_channels==32),"invalid LN config");
    c10::cuda::CUDAGuard guard(x.device());int m=x.size(0),d=x.size(1);
    int grid=std::max(1,std::min((m+warps-1)/warps,at::cuda::getDeviceProperties(x.get_device())->multiProcessorCount*waves));
    auto dx=torch::empty_like(x);auto dg=torch::empty_like(w),db=torch::empty_like(w);
    auto pg=torch::empty({grid*warps,d},w.options()),pb=torch::empty_like(pg);auto stream=at::cuda::getCurrentCUDAStream();
    #define BWD(D,T) initialize_kernel(norm_backward<D,T>); norm_backward<D,T><<<grid,32*warps,0,stream>>>((BF*)dy.data_ptr(),(BF*)x.data_ptr(),w.data_ptr<float>(),mean.data_ptr<float>(),rs.data_ptr<float>(),(BF*)residual.data_ptr(),(BF*)dx.data_ptr(),pg.data_ptr<float>(),pb.data_ptr<float>(),m)
    #define DISPATCH(D) case D: if(tx==8){BWD(D,8);}else{BWD(D,16);}break;
    switch(d){DISPATCH(128) DISPATCH(256) DISPATCH(384) DISPATCH(512)}
    #undef BWD
    #undef DISPATCH
    initialize_kernel(norm_reduce,2*256*sizeof(float));  // largest supported reduction scratch
    norm_reduce<<<(d+reduce_channels-1)/reduce_channels,reduce_threads,2*(reduce_threads/32)*reduce_channels*sizeof(float),stream>>>(pg.data_ptr<float>(),pb.data_ptr<float>(),dg.data_ptr<float>(),db.data_ptr<float>(),grid*warps,d,reduce_channels);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"LN backward launch failed");return {dx,dg,db};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);}
