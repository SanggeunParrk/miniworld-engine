// CTA-wide row reductions for widths that overfill a single warp's registers.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>
template<class A> __device__ A warp_sum(A v){for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return __shfl_sync(0xffffffff,v,0);}
template<class A,int T> __device__ A block_sum(A v,A* shared){
 v=warp_sum(v);if(threadIdx.x%32==0)shared[threadIdx.x/32]=v;__syncthreads();
 v=threadIdx.x<T/32?shared[threadIdx.x]:A(0);v=warp_sum(v);if(threadIdx.x==0)shared[0]=v;__syncthreads();v=shared[0];__syncthreads();return v;
}
template<class A> __device__ A invsqrt(A x){return A(1)/sqrt(x);}
template<> __device__ float invsqrt(float x){return rsqrtf(x);}
template<class S,class A,int T,int C,bool RMS>
__global__ void wide_forward(const S* x,const A* w,const A* b,S* y,A* mean,A* inv,int64_t M,int D,A eps){
 __shared__ A scratch[T/32];int64_t row=blockIdx.x;A xv[C],mu=0,var=0;
 #pragma unroll
 for(int k=0;k<C;++k){int c=threadIdx.x+k*T;xv[k]=c<D?A(x[row*D+c]):A(0);if(!RMS)mu+=xv[k];}
 if(!RMS)mu=block_sum<A,T>(mu,scratch)/D;
 #pragma unroll
 for(int k=0;k<C;++k){int c=threadIdx.x+k*T;if(c<D){A z=xv[k]-mu;var+=z*z;}}
 A rs=invsqrt(block_sum<A,T>(var,scratch)/D+eps);if(threadIdx.x==0){mean[row]=mu;inv[row]=rs;}
 #pragma unroll
 for(int k=0;k<C;++k){int c=threadIdx.x+k*T;if(c<D)y[row*D+c]=S((xv[k]-mu)*rs*(w?w[c]:A(1))+(b?b[c]:A(0)));}
}
template<class S,class A,int T,int C,bool RMS>
__global__ void wide_backward(const S* x,const S* dy,const A* w,const A* mean,const A* inv,S* dx,A* pw,A* pb,int64_t M,int D,int P){
 __shared__ A scratch[T/32];A sw[C]={},sb[C]={};int group=blockIdx.x;
 for(int64_t row=group;row<M;row+=P){
  A z[C],g[C],s1=0,s2=0,r=inv[row],mu=RMS?A(0):mean[row];
  #pragma unroll
  for(int k=0;k<C;++k){int c=threadIdx.x+k*T;A v=c<D?A(dy[row*D+c]):A(0);z[k]=c<D?(A(x[row*D+c])-mu)*r:A(0);g[k]=v*(w&&c<D?w[c]:A(1));if(!RMS)s1+=g[k];s2+=g[k]*z[k];if(pw)sw[k]+=v*z[k];if(pb)sb[k]+=v;}
  if(!RMS)s1=block_sum<A,T>(s1,scratch)/D;s2=block_sum<A,T>(s2,scratch)/D;
  #pragma unroll
  for(int k=0;k<C;++k){int c=threadIdx.x+k*T;if(c<D)dx[row*D+c]=S((g[k]-(RMS?A(0):s1)-z[k]*s2)*r);}
 }
 #pragma unroll
 for(int k=0;k<C;++k){int c=threadIdx.x+k*T;if(c<D){if(pw)pw[int64_t(group)*D+c]=sw[k];if(pb)pb[int64_t(group)*D+c]=sb[k];}}
}
// 8 columns per CTA, 32 lanes reduce independent partial rows. Coalesced groups
// share each cache line, avoiding one whole CTA and strided transactions per column.
template<class A> __global__ void finish(const A* pw,const A* pb,A* dw,A* db,int P,int D){
 int c=blockIdx.x*8+threadIdx.x%8,r=threadIdx.x/8;A sw=0,sb=0;
 for(int p=r;p<P;p+=32)if(c<D){if(pw)sw+=pw[int64_t(p)*D+c];if(pb)sb+=pb[int64_t(p)*D+c];}
 __shared__ A ws[256],bs[256];ws[threadIdx.x]=sw;bs[threadIdx.x]=sb;__syncthreads();
 for(int n=16;n;n>>=1){if(r<n){ws[threadIdx.x]+=ws[threadIdx.x+n*8];bs[threadIdx.x]+=bs[threadIdx.x+n*8];}__syncthreads();}
 if(r==0&&c<D){if(dw)dw[c]=ws[threadIdx.x];if(db)db[c]=bs[threadIdx.x];}
}
void validate(torch::Tensor x,torch::Tensor w,torch::Tensor b){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.dim()>=1&&x.size(-1)>0&&x.size(-1)<=16384,"wide norm expects CUDA contiguous width <=16384");auto acc=x.scalar_type()==at::kDouble?at::kDouble:at::kFloat;
 for(auto p:{w,b})if(p.defined())TORCH_CHECK(p.device()==x.device()&&p.scalar_type()==acc&&p.dim()==1&&p.numel()==x.size(-1)&&p.is_contiguous(),"invalid affine");
}
#define FW(T,C,R) wide_forward<scalar_t,A,T,C,R><<<M,T,0,stream>>>(x.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,b.defined()?b.data_ptr<A>():nullptr,y.data_ptr<scalar_t>(),mean.data_ptr<A>(),inv.data_ptr<A>(),M,D,A(eps))
#define FWR(T,C) if(rms){FW(T,C,true);}else{FW(T,C,false);}
#define FWT(T) if(D<=T*8){FWR(T,8)}else if(D<=T*16){FWR(T,16)}else if(D<=T*32){FWR(T,32)}else if(D<=T*64){FWR(T,64)}else{FWR(T,128)}
std::vector<torch::Tensor> forward(torch::Tensor x,c10::optional<torch::Tensor> weight,c10::optional<torch::Tensor> bias,double eps,bool rms,int threads){
 auto w=weight.value_or(torch::Tensor()),b=bias.value_or(torch::Tensor());validate(x,w,b);TORCH_CHECK(threads==128||threads==256,"invalid threads");c10::cuda::CUDAGuard guard(x.device());int D=x.size(-1);int64_t M=x.numel()/D;auto opt=x.options().dtype(x.scalar_type()==at::kDouble?at::kDouble:at::kFloat);auto y=torch::empty_like(x),mean=torch::empty({M},opt),inv=torch::empty({M},opt);
 if(M)AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"wide_forward",[&]{using A=std::conditional_t<std::is_same_v<scalar_t,double>,double,float>;auto stream=at::cuda::getCurrentCUDAStream();if(threads==128){FWT(128)}else{FWT(256)}});
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,mean,inv};
}
#define BW(T,C,R) wide_backward<scalar_t,A,T,C,R><<<P,T,0,stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,mean.data_ptr<A>(),inv.data_ptr<A>(),dx.data_ptr<scalar_t>(),w.defined()?pw.data_ptr<A>():nullptr,has_bias?pb.data_ptr<A>():nullptr,M,D,P)
#define BWR(T,C) if(rms){BW(T,C,true);}else{BW(T,C,false);}
#define BWT(T) if(D<=T*8){BWR(T,8)}else if(D<=T*16){BWR(T,16)}else if(D<=T*32){BWR(T,32)}else if(D<=T*64){BWR(T,64)}else{BWR(T,128)}
std::vector<torch::Tensor> backward(torch::Tensor x,torch::Tensor dy,c10::optional<torch::Tensor> weight,torch::Tensor mean,torch::Tensor inv,bool has_bias,bool rms,int threads,int rows){
 auto w=weight.value_or(torch::Tensor());validate(x,w,torch::Tensor());TORCH_CHECK(dy.sizes()==x.sizes()&&dy.scalar_type()==x.scalar_type()&&dy.device()==x.device()&&dy.is_contiguous(),"invalid gradient");TORCH_CHECK((threads==128||threads==256)&&rows>=1&&rows<=4096,"invalid config");c10::cuda::CUDAGuard guard(x.device());int D=x.size(-1);int64_t M=x.numel()/D;auto opt=x.options().dtype(x.scalar_type()==at::kDouble?at::kDouble:at::kFloat);
 TORCH_CHECK(mean.device()==x.device()&&inv.device()==x.device()&&mean.numel()==M&&inv.numel()==M&&mean.scalar_type()==opt.dtype().toScalarType()&&inv.scalar_type()==mean.scalar_type()&&mean.is_contiguous()&&inv.is_contiguous(),"invalid statistics");
 int64_t cap=std::max<int64_t>(1,(32*1024*1024)/(2*D*mean.element_size()));int P=std::min<int64_t>((M+rows-1)/rows,cap);auto dx=torch::empty_like(x),dw=torch::empty({w.defined()?D:0},opt),db=torch::empty({has_bias?D:0},opt),pw=torch::empty({w.defined()?P:0,D},opt),pb=torch::empty({has_bias?P:0,D},opt);
 AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"wide_backward",[&]{using A=std::conditional_t<std::is_same_v<scalar_t,double>,double,float>;auto stream=at::cuda::getCurrentCUDAStream();if(M){if(threads==128){BWT(128)}else{BWT(256)}}if(w.defined()||has_bias)finish<A><<<(D+7)/8,256,0,stream>>>(w.defined()?pw.data_ptr<A>():nullptr,has_bias?pb.data_ptr<A>():nullptr,w.defined()?dw.data_ptr<A>():nullptr,has_bias?db.data_ptr<A>():nullptr,P,D);});C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dw,db};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);}
