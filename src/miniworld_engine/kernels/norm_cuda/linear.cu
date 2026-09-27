// LN + small/medium output GEMM. One CTA owns each row tile, so saved activations
// have a single writer. Tensor Core operands preserve the standalone LN rounding.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
template<class T> __device__ float readf(T x){return float(x);}
template<> __device__ float readf(__half x){return __half2float(x);}
template<> __device__ float readf(__nv_bfloat16 x){return __bfloat162float(x);}
template<class T> __device__ T writef(float x);
template<> __device__ __half writef(float x){return __float2half_rn(x);}
template<> __device__ __nv_bfloat16 writef(float x){return __float2bfloat16_rn(x);}
__device__ float rowsum(float x){for(int o=4;o;o>>=1)x+=__shfl_down_sync(0xffffffff,x,o,8);return __shfl_sync(0xffffffff,x,0,8);}
template<class T,int V> struct alignas(sizeof(T)*V) Pack{T a[V];};
template<class T,int K,int BM,int NT,int Threads>
__global__ void ln_linear(const T* x,const float* gamma,const float* beta,const T* w,const T* bias,
 T* y,T* xn,float* mean,float* inv,int64_t M,int N,float eps) {
 extern __shared__ __align__(32) unsigned char mem[];
 constexpr int S=K+8;
 T* sn=reinterpret_cast<T*>(mem);
 T* sw=sn+BM*S;
 float* out=reinterpret_cast<float*>(sw+NT*S);
 int lane=threadIdx.x%32,warp=threadIdx.x/32,nlane=threadIdx.x%8;
 int64_t start=int64_t(blockIdx.x)*BM;
 for(int r=threadIdx.x/8;r<BM;r+=Threads/8){
  float vals[K/8],mu=0,var=0;
  #pragma unroll
  for(int j=0;j<K/8;j+=4){
   int c=(j/4*8+nlane)*4;Pack<T,4> q;
   if(start+r<M && uintptr_t(x)%8==0)q=*reinterpret_cast<const Pack<T,4>*>(x+(start+r)*K+c);
   #pragma unroll
   for(int u=0;u<4;++u){vals[j+u]=start+r<M?(uintptr_t(x)%8==0?readf(q.a[u]):readf(x[(start+r)*K+c+u])):0.f;mu+=vals[j+u];}
  }
  mu=rowsum(mu)/K;
  #pragma unroll
  for(int j=0;j<K/8;++j){float v=vals[j]-mu;var+=v*v;}
  float rs=rsqrtf(rowsum(var)/K+eps);
  if(nlane==0 && start+r<M){mean[start+r]=mu;inv[start+r]=rs;}
  #pragma unroll
  for(int j=0;j<K/8;j+=4){
   int c=(j/4*8+nlane)*4;Pack<T,4> q;
   #pragma unroll
   for(int u=0;u<4;++u)q.a[u]=writef<T>((vals[j+u]-mu)*rs*gamma[c+u]+beta[c+u]);
   *reinterpret_cast<Pack<T,4>*>(sn+r*S+c)=q;
   if(xn&&start+r<M)*reinterpret_cast<Pack<T,4>*>(xn+(start+r)*K+c)=q;
  }

 }
 __syncthreads();
 for(int nb=0;nb<N;nb+=NT){
  int tile_n=min(NT,N-nb);
  for(int v=threadIdx.x;v<tile_n*(K/8);v+=blockDim.x){int nr=v/(K/8),kc=v%(K/8)*8;*reinterpret_cast<uint4*>(sw+nr*S+kc)=*reinterpret_cast<const uint4*>(w+(nb+nr)*K+kc);}
  __syncthreads();
  for(int work=warp;work<(BM/16)*(tile_n/16);work+=Threads/32){
    int mr=(work/(tile_n/16))*16;
    int nc=(work%(tile_n/16))*16;
    int n=nb+nc;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    #pragma unroll
    for(int k=0;k<K;k+=16){
     wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> af;
     wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> bf;
     wmma::load_matrix_sync(af,sn+mr*S+k,S);
     wmma::load_matrix_sync(bf,sw+nc*S+k,S);
     wmma::mma_sync(acc,af,bf,acc);
    }
    wmma::store_matrix_sync(out+warp*256,acc,16,wmma::mem_row_major);
    __syncwarp();
    for(int q=lane;q<256;q+=32){int r=q/16,c=q%16;int64_t row=start+mr+r;if(row<M)y[row*N+n+c]=writef<T>(out[warp*256+q]+(bias?readf(bias[n+c]):0.f));}
    __syncwarp();
  }
  __syncthreads();
 }

}
template<class T,int K,int BM,int NT,int Threads>
void launch_shape(torch::Tensor x,torch::Tensor gamma,torch::Tensor beta,torch::Tensor w,torch::Tensor bias,torch::Tensor y,torch::Tensor xn,torch::Tensor mean,torch::Tensor inv,double eps,bool save){
 int N=w.size(0);int64_t M=x.numel()/K;int grid=(M+BM-1)/BM;int smem=(BM+NT)*(K+8)*sizeof(T)+(Threads/32)*256*sizeof(float);auto stream=at::cuda::getCurrentCUDAStream();
 static thread_local int configured_device=-1;
 if(smem>48*1024 && configured_device!=x.get_device()){C10_CUDA_CHECK(cudaFuncSetAttribute(ln_linear<T,K,BM,NT,Threads>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));configured_device=x.get_device();}
 ln_linear<T,K,BM,NT,Threads><<<grid,Threads,smem,stream>>>(reinterpret_cast<const T*>(x.data_ptr()),gamma.data_ptr<float>(),beta.data_ptr<float>(),reinterpret_cast<const T*>(w.data_ptr()),bias.defined()?reinterpret_cast<const T*>(bias.data_ptr()):nullptr,reinterpret_cast<T*>(y.data_ptr()),save?reinterpret_cast<T*>(xn.data_ptr()):nullptr,mean.data_ptr<float>(),inv.data_ptr<float>(),M,N,float(eps));
}
#define LS(K,BM,NT,T) launch_shape<Tscalar,K,BM,NT,T>(x,gamma,beta,w,bias,y,xn,mean,inv,eps,save)
#define LAUNCH(K) if(w.size(0)<=32){LS(K,64,32,128);}else if(w.size(0)>=128){LS(K,32,128,256);}else{LS(K,32,64,128);}
template<class Tscalar> void launch(torch::Tensor x,torch::Tensor gamma,torch::Tensor beta,torch::Tensor w,torch::Tensor bias,torch::Tensor y,torch::Tensor xn,torch::Tensor mean,torch::Tensor inv,double eps,bool save){
 switch(x.size(-1)){case 64:LAUNCH(64);break;case 128:LAUNCH(128);break;case 256:LAUNCH(256);break;case 384:LAUNCH(384);break;case 512:LAUNCH(512);break;default:TORCH_CHECK(false,"unsupported fused input width");}
}
std::vector<torch::Tensor> linear_forward(torch::Tensor x,torch::Tensor gamma,torch::Tensor beta,torch::Tensor w,c10::optional<torch::Tensor> bias_opt,double eps,bool save){
 auto bias=bias_opt.value_or(torch::Tensor());
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.dim()==2,"expected contiguous CUDA matrix");
 TORCH_CHECK(x.scalar_type()==at::kHalf||x.scalar_type()==at::kBFloat16,"fused linear requires FP16/BF16");
 int K=x.size(1);TORCH_CHECK(K==64||K==128||K==256||K==384||K==512,"unsupported fused width");
 TORCH_CHECK(w.device()==x.device()&&w.scalar_type()==x.scalar_type()&&w.dim()==2&&w.size(1)==K&&w.size(0)>0&&w.size(0)%16==0&&w.is_contiguous(),"invalid fused matrix weight");
 for(auto p:{gamma,beta})TORCH_CHECK(p.device()==x.device()&&p.scalar_type()==at::kFloat&&p.dim()==1&&p.numel()==K&&p.is_contiguous(),"invalid norm parameters");
 if(bias.defined())TORCH_CHECK(bias.device()==x.device()&&bias.scalar_type()==x.scalar_type()&&bias.dim()==1&&bias.numel()==w.size(0)&&bias.is_contiguous(),"invalid linear bias");
 // Only the matrix weight is loaded by WMMA directly from global memory.
 if(uintptr_t(w.data_ptr())%32)w=w.clone();
 c10::cuda::CUDAGuard guard(x.device());auto f=x.options().dtype(at::kFloat);int64_t M=x.size(0);auto y=torch::empty({M,w.size(0)},x.options()),xn=torch::empty({save?M:0,K},x.options()),mean=torch::empty({M},f),inv=torch::empty({M},f);
 if(M){if(x.scalar_type()==at::kHalf)launch<__half>(x,gamma,beta,w,bias,y,xn,mean,inv,eps,save);else launch<__nv_bfloat16>(x,gamma,beta,w,bias,y,xn,mean,inv,eps,save);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,xn,mean,inv};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&linear_forward);}
