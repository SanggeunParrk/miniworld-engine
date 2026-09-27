// Generic last-axis normalization. No atomics: bounded FP32/FP64 partials for
// affine gradients, then a deterministic column reduction. Warp owns a row.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>

template<class A,int W=32> __device__ A sumwarp(A v) {
  unsigned mask=(0xffffffffu >> (32-W)) << ((threadIdx.x%32)/W*W);
  for (int d=W/2; d; d>>=1) v += __shfl_down_sync(mask, v, d, W);
  return __shfl_sync(mask, v, 0, W);
}
template<class A> __device__ A invsqrt(A v) { return A(1)/sqrt(v); }
template<> __device__ float invsqrt(float v) { return rsqrtf(v); }
template<class T, class A, bool RMS>
__global__ void forward_norm(const T* x, const A* w, const A* b, T* y,
                            A* mean, A* inv, int64_t M, int D, A eps) {
  int lane=threadIdx.x%32;
  int64_t row=int64_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
  for(;row<M;row+=int64_t(gridDim.x)*(blockDim.x/32)) {
  A mu=0;
  if(!RMS) { for(int c=lane;c<D;c+=32) mu+=A(x[row*D+c]); mu=sumwarp(mu)/D; }
  A var=0;
  for(int c=lane;c<D;c+=32) {A z=A(x[row*D+c])-mu;var+=z*z;}
  A r=invsqrt(sumwarp(var)/D+eps);
  if(lane==0) {mean[row]=mu;inv[row]=r;}
  for(int c=lane;c<D;c+=32) {
    A z=(A(x[row*D+c])-mu)*r;
    y[row*D+c]=T(z*(w?w[c]:A(1))+(b?b[c]:A(0)));
  }
  }
}

template<class T,int V> struct alignas(sizeof(T)*V) Packed {T a[V];};
template<class T,class A,bool RMS,int C,int V,int Fixed=0,int W=32>
__global__ void forward_vec(const T* x,const A* w,const A* b,T* y,A* mean,A* inv,int64_t M,int width,A eps) {
 const int D=Fixed?Fixed:width;
 int lane=threadIdx.x%W;
 for(int64_t row=int64_t(blockIdx.x)*(blockDim.x/W)+threadIdx.x/W;row<M;row+=int64_t(gridDim.x)*(blockDim.x/W)) {
  A xv[C]={},mu=0,var=0;
  #pragma unroll
  for(int k=0;k<C;k+=V){
   int c=(k/V*W+lane)*V;
   if(c<D){auto q=*reinterpret_cast<const Packed<T,V>*>(x+row*D+c);
    #pragma unroll
    for(int j=0;j<V;++j){xv[k+j]=A(q.a[j]);if(!RMS)mu+=xv[k+j];}
   }
  }
  if(!RMS)mu=sumwarp<A,W>(mu)/D;
  #pragma unroll
  for(int k=0;k<C;++k){int c=(k/V*W+lane)*V+k%V;if(c<D){A z=xv[k]-mu;var+=z*z;}}
  A r=invsqrt(sumwarp<A,W>(var)/D+eps);
  if(lane==0){mean[row]=mu;inv[row]=r;}
  #pragma unroll
  for(int k=0;k<C;k+=V){
   int c=(k/V*W+lane)*V;
   if(c<D){Packed<T,V> out;Packed<A,V> wp,bp;
    if(w)wp=*reinterpret_cast<const Packed<A,V>*>(w+c);
    if(b)bp=*reinterpret_cast<const Packed<A,V>*>(b+c);
    #pragma unroll
    for(int j=0;j<V;++j)out.a[j]=T((xv[k+j]-mu)*r*(w?wp.a[j]:A(1))+(b?bp.a[j]:A(0)));
    *reinterpret_cast<Packed<T,V>*>(y+row*D+c)=out;
   }
  }
 }
}
// Width slices of 32 are revisited after the two row reductions. A warp scans
// ROWS rows; affine partials are assigned to a second column kernel to avoid
// storing D-sized register arrays (arbitrary D and predictable register usage).
template<class T,class A,bool RMS>
__global__ void backward_dx(const T* x,const T* dy,const A* w,const A* mean,
                           const A* inv,T* dx,int64_t M,int D) {
  int lane=threadIdx.x%32;
  int64_t row=int64_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
  if(row>=M) return;
  A mu=mean[row],r=inv[row],s1=0,s2=0;
  for(int c=lane;c<D;c+=32) {
    A z=(A(x[row*D+c])-mu)*r;
    A g=A(dy[row*D+c])*(w?w[c]:A(1));
    s1+=g;s2+=g*z;
  }
  if(!RMS) s1=sumwarp(s1)/D;
  s2=sumwarp(s2)/D;
  for(int c=lane;c<D;c+=32) {
    A z=(A(x[row*D+c])-mu)*r;
    A g=A(dy[row*D+c])*(w?w[c]:A(1));
    dx[row*D+c]=T((g-(RMS?A(0):s1)-z*s2)*r);
  }
}
template<class T,class A>
__global__ void affine_partial(const T* x,const T* dy,const A* mean,const A* inv,
                              A* pw,A* pb,int64_t M,int D,int rows) {
  int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=D) return;
  int64_t first=int64_t(blockIdx.y)*rows,last=min(first+rows,M);
  A sw=0,sb=0;
  for(int64_t row=first;row<last;++row) {
    A g=A(dy[row*D+c]);
    if(pw) sw+=g*(A(x[row*D+c])-mean[row])*inv[row];
    if(pb) sb+=g;
  }
  if(pw) pw[int64_t(blockIdx.y)*D+c]=sw;
  if(pb) pb[int64_t(blockIdx.y)*D+c]=sb;
}

// Persistent warp owns strided rows. Vector loads support FP32 affine parameters
// independently of activation dtype; weight values remain resident across rows.
template<class T,class A,bool RMS,int C,int V,int Fixed=0,int W=32>
__global__ void backward_fused(const T* x,const T* dy,const A* w,const A* mean,
 const A* inv,T* dx,A* pw,A* pb,int64_t M,int width,int rows,int P) {
  const int D=Fixed?Fixed:width;
  int lane=threadIdx.x%W;
  int group=blockIdx.x*(blockDim.x/W)+threadIdx.x/W;
  extern __shared__ __align__(16) unsigned char partial_bytes[];
  A* partial=reinterpret_cast<A*>(partial_bytes);
  A sw[C]={},sb[C]={},wv[C];
  #pragma unroll
  for(int k=0;k<C;k+=V) {
    int c=(k/V*W+lane)*V;
    Packed<A,V> q;
    if(w&&c<D)q=*reinterpret_cast<const Packed<A,V>*>(w+c);
    #pragma unroll
    for(int j=0;j<V;++j)wv[k+j]=w&&c<D?q.a[j]:A(1);
  }
  for(int64_t row=group;group<P&&row<M;row+=P) {
    A z[C]={},g[C]={},s1=0,s2=0,mu=RMS?A(0):mean[row],r=inv[row];
    #pragma unroll
    for(int k=0;k<C;k+=V) {
      int c=(k/V*W+lane)*V;
      if(c<D) {
        auto xv=*reinterpret_cast<const Packed<T,V>*>(x+row*D+c);
        auto gv=*reinterpret_cast<const Packed<T,V>*>(dy+row*D+c);
        #pragma unroll
        for(int j=0;j<V;++j) {
          A v=A(gv.a[j]); z[k+j]=(A(xv.a[j])-mu)*r;
          g[k+j]=v*wv[k+j];
          if(!RMS)s1+=g[k+j];s2+=g[k+j]*z[k+j];
          if(pw)sw[k+j]+=v*z[k+j];if(pb)sb[k+j]+=v;
        }
      }
    }
    if(!RMS)s1=sumwarp<A,W>(s1)/D;
    s2=sumwarp<A,W>(s2)/D;
    #pragma unroll
    for(int k=0;k<C;k+=V) {
      int c=(k/V*W+lane)*V;
      if(c<D) {
        Packed<T,V> out;
        #pragma unroll
        for(int j=0;j<V;++j)out.a[j]=T((g[k+j]-(RMS?A(0):s1)-z[k+j]*s2)*r);
        *reinterpret_cast<Packed<T,V>*>(dx+row*D+c)=out;
      }
    }
  }
  // Reduce row groups within a warp, then across the CTA. Only one affine
  // partial per CTA reaches global memory, regardless of subgroup row count.
  int warp=threadIdx.x/32;
  #pragma unroll
  for(int k=0;k<C;++k){
    #pragma unroll
    for(int off=16;off>=W;off>>=1){sw[k]+=__shfl_down_sync(0xffffffff,sw[k],off);sb[k]+=__shfl_down_sync(0xffffffff,sb[k],off);}
    int c=(k/V*W+lane)*V+k%V;
    if(threadIdx.x%32<W&&c<D){if(pw)partial[warp*D+c]=sw[k];if(pb)partial[(blockDim.x/32+warp)*D+c]=sb[k];}
  }
  __syncthreads();
  for(int c=threadIdx.x;c<D;c+=blockDim.x){A a=0,b=0;for(int q=0;q<blockDim.x/32;++q){if(pw)a+=partial[q*D+c];if(pb)b+=partial[(blockDim.x/32+q)*D+c];}if(pw)pw[int64_t(blockIdx.x)*D+c]=a;if(pb)pb[int64_t(blockIdx.x)*D+c]=b;}

}
template<class T,class A,bool RMS,int C,int V,int Fixed=0>
__global__ void backward_warp(const T* x,const T* dy,const A* w,const A* mean,
 const A* inv,T* dx,A* pw,A* pb,int64_t M,int width,int rows,int P) {
  const int D=Fixed?Fixed:width;
  int lane=threadIdx.x%32;
  int group=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if(group>=P)return;
  A sw[C]={},sb[C]={},wv[C];
  #pragma unroll
  for(int k=0;k<C;k+=V) {
    int c=(k/V*32+lane)*V;
    Packed<A,V> q;
    if(w&&c<D)q=*reinterpret_cast<const Packed<A,V>*>(w+c);
    #pragma unroll
    for(int j=0;j<V;++j)wv[k+j]=w&&c<D?q.a[j]:A(1);
  }
  for(int64_t row=group;row<M;row+=P) {
    A z[C]={},g[C]={},s1=0,s2=0,mu=RMS?A(0):mean[row],r=inv[row];
    #pragma unroll
    for(int k=0;k<C;k+=V) {
      int c=(k/V*32+lane)*V;
      if(c<D) {
        auto xv=*reinterpret_cast<const Packed<T,V>*>(x+row*D+c);
        auto gv=*reinterpret_cast<const Packed<T,V>*>(dy+row*D+c);
        #pragma unroll
        for(int j=0;j<V;++j) {
          A v=A(gv.a[j]); z[k+j]=(A(xv.a[j])-mu)*r;
          g[k+j]=v*wv[k+j];
          if(!RMS)s1+=g[k+j];s2+=g[k+j]*z[k+j];
          if(pw)sw[k+j]+=v*z[k+j];if(pb)sb[k+j]+=v;
        }
      }
    }
    if(!RMS)s1=sumwarp(s1)/D;
    s2=sumwarp(s2)/D;
    #pragma unroll
    for(int k=0;k<C;k+=V) {
      int c=(k/V*32+lane)*V;
      if(c<D) {
        Packed<T,V> out;
        #pragma unroll
        for(int j=0;j<V;++j)out.a[j]=T((g[k+j]-(RMS?A(0):s1)-z[k+j]*s2)*r);
        *reinterpret_cast<Packed<T,V>*>(dx+row*D+c)=out;
      }
    }
  }
  #pragma unroll
  for(int k=0;k<C;k+=V) {
    int c=(k/V*32+lane)*V;
    if(c<D) {
      Packed<A,V> qw,qb;
      #pragma unroll
      for(int j=0;j<V;++j){qw.a[j]=sw[k+j];qb.a[j]=sb[k+j];}
      if(pw)*reinterpret_cast<Packed<A,V>*>(pw+int64_t(group)*D+c)=qw;
      if(pb)*reinterpret_cast<Packed<A,V>*>(pb+int64_t(group)*D+c)=qb;
    }
  }
}
template<class A>
__global__ void affine_finish(const A* pw,const A* pb,A* dw,A* db,int P,int D) {
  int c=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
  A sw=0,sb=0;
  for(int p=threadIdx.x;p<P;p+=blockDim.x) {if(pw)sw+=pw[int64_t(p)*D+c];if(pb)sb+=pb[int64_t(p)*D+c];}
  sw=sumwarp(sw);sb=sumwarp(sb);
  __shared__ A ws[8],bs[8];
  if(lane==0){ws[warp]=sw;bs[warp]=sb;}__syncthreads();
  if(warp==0) {
    sw=sumwarp(lane<8?ws[lane]:A(0));sb=sumwarp(lane<8?bs[lane]:A(0));
    if(lane==0){if(dw)dw[c]=sw;if(db)db[c]=sb;}
  }
}
void validate(torch::Tensor x,torch::Tensor w,torch::Tensor b) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim()>=1,"contiguous CUDA input required");
  TORCH_CHECK(x.size(-1)>0 && x.size(-1)<=INT_MAX,"invalid normalized width");
  auto acc=x.scalar_type()==at::kDouble?at::kDouble:at::kFloat;
  for(auto p:{w,b}) if(p.defined()) {
    TORCH_CHECK(p.device()==x.device() && p.scalar_type()==acc && p.is_contiguous() && p.dim()==1 && p.numel()==x.size(-1),"invalid affine parameter");
  }
}
#define FV(C,V,F,R) forward_vec<scalar_t,A,R,C,V,F,(F==128||F==64)?8:32><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,b.defined()?b.data_ptr<A>():nullptr,y.data_ptr<scalar_t>(),mean.data_ptr<A>(),inv.data_ptr<A>(),M,D,A(eps))
#define FPICK(C,V,F) if(rms){FV(C,V,F,true);}else{FV(C,V,F,false);}
std::vector<torch::Tensor> norm_forward(torch::Tensor x,c10::optional<torch::Tensor> weight,
 c10::optional<torch::Tensor> bias,double eps,bool rms,int threads) {
  torch::Tensor w=weight.value_or(torch::Tensor()),b=bias.value_or(torch::Tensor());validate(x,w,b);
  TORCH_CHECK(threads==128||threads==256,"invalid block size");
  c10::cuda::CUDAGuard guard(x.device());
  auto opt=x.options().dtype(x.scalar_type()==at::kDouble?at::kDouble:at::kFloat);
  int D=x.size(-1);int64_t M=x.numel()/D;
  auto y=torch::empty_like(x),mean=torch::empty({M},opt),inv=torch::empty({M},opt);
  if(M) AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"norm_forward",[&]{
    using A=std::conditional_t<std::is_same_v<scalar_t,double>,double,float>;
    auto stream=at::cuda::getCurrentCUDAStream();
    int64_t grid=std::min<int64_t>((M+threads/32-1)/(threads/32),32*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    auto aligned=[&](int v){return (D%(32*v)==0)&&uintptr_t(x.data_ptr())%(sizeof(scalar_t)*v)==0&&(!w.defined()||uintptr_t(w.data_ptr())%(sizeof(A)*v)==0)&&(!b.defined()||uintptr_t(b.data_ptr())%(sizeof(A)*v)==0);};
    if(D<=1024&&aligned(4)) {
      if(D==128){grid=std::min<int64_t>((M+threads/8-1)/(threads/8),32*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);FPICK(16,4,128)}else if(D==256){FPICK(8,4,256)}else if(D==384){FPICK(12,4,384)}else if(D==512){FPICK(16,4,512)}else if(D==768){FPICK(24,4,768)}else if(D==1024){FPICK(32,4,1024)}else{FPICK(32,4,0)}
    } else if(D==64&&aligned(2)) {grid=std::min<int64_t>((M+threads/8-1)/(threads/8),32*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);FPICK(8,2,64)}
    else if(rms) forward_norm<scalar_t,A,true><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,b.defined()?b.data_ptr<A>():nullptr,y.data_ptr<scalar_t>(),mean.data_ptr<A>(),inv.data_ptr<A>(),M,D,A(eps));
    else forward_norm<scalar_t,A,false><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,b.defined()?b.data_ptr<A>():nullptr,y.data_ptr<scalar_t>(),mean.data_ptr<A>(),inv.data_ptr<A>(),M,D,A(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,mean,inv};
}
#define LAUNCH(C,V,F,R) backward_fused<scalar_t,A,R,C,V,F,(F==128||F==64)?8:32><<<grid,threads,2*(threads/32)*D*sizeof(A),stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,mean.data_ptr<A>(),inv.data_ptr<A>(),dx.data_ptr<scalar_t>(),w.defined()?pw.data_ptr<A>():nullptr,has_bias?pb.data_ptr<A>():nullptr,M,D,rows,P)
#define OLDLAUNCH(C,V,F,R) backward_warp<scalar_t,A,R,C,V,F><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,mean.data_ptr<A>(),inv.data_ptr<A>(),dx.data_ptr<scalar_t>(),w.defined()?pw.data_ptr<A>():nullptr,has_bias?pb.data_ptr<A>():nullptr,M,D,rows,P)
#define OLDWIDTH(C,V,F) if(rms){OLDLAUNCH(C,V,F,true);}else{OLDLAUNCH(C,V,F,false);}
#define WIDTH(C,V,F) if(rms){LAUNCH(C,V,F,true);}else{LAUNCH(C,V,F,false);}
#define BWIDTH(V) if(D<=128){WIDTH(4,V,0)}else if(D<=256){WIDTH(8,V,0)}else if(D<=512){WIDTH(16,V,0)}else{WIDTH(32,V,0)}
std::vector<torch::Tensor> norm_backward(torch::Tensor x,torch::Tensor dy,
 c10::optional<torch::Tensor> weight,torch::Tensor mean,torch::Tensor inv,
 bool has_bias,bool rms,int threads,int rows) {
  torch::Tensor w=weight.value_or(torch::Tensor());validate(x,w,torch::Tensor());
  TORCH_CHECK(dy.sizes()==x.sizes() && dy.scalar_type()==x.scalar_type() && dy.device()==x.device() && dy.is_contiguous(),"invalid dy");
  TORCH_CHECK((threads==128||threads==256)&&rows>=1&&rows<=4096,"invalid launch configuration");
  c10::cuda::CUDAGuard guard(x.device());
  int D=x.size(-1);int64_t M=x.numel()/D;
  int64_t max_groups=std::max<int64_t>(1,std::min<int64_t>(65535,(32*1024*1024)/(2*D*mean.element_size())));
  rows=std::max<int64_t>(rows,(M+max_groups-1)/max_groups);
  int P=(M+rows-1)/rows;int partials=P;
  auto dx=torch::empty_like(x),dw=torch::empty({w.defined()?D:0},mean.options()),db=torch::empty({has_bias?D:0},mean.options());
  auto aligned_host=[&](int v){return uintptr_t(x.data_ptr())%(x.element_size()*v)==0 && uintptr_t(dy.data_ptr())%(x.element_size()*v)==0 && (!w.defined()||uintptr_t(w.data_ptr())%(mean.element_size()*v)==0);};
  bool narrow=(D==128&&aligned_host(4))||(D==64&&aligned_host(2));
  bool medium=(D==256||D==384||D==512)&&aligned_host(4)&&2*(threads/32)*D*mean.element_size()<=48*1024;
  int groups_per_cta=narrow?threads/8:threads/32;
  int partial_slots=(narrow||medium)?(P+groups_per_cta-1)/groups_per_cta:P;
  auto pw=torch::empty({w.defined()?partial_slots:0,D},mean.options()),pb=torch::empty({has_bias?partial_slots:0,D},mean.options());
  TORCH_CHECK(mean.device()==x.device()&&inv.device()==x.device()&&mean.numel()==M&&inv.numel()==M&&mean.is_contiguous()&&inv.is_contiguous(),"invalid statistics");
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"norm_backward",[&]{
    using A=std::conditional_t<std::is_same_v<scalar_t,double>,double,float>;
    auto stream=at::cuda::getCurrentCUDAStream();
    if(M && D<=1024) {
      int grid=(P+threads/32-1)/(threads/32);
      auto aligned=[&](int v){return D%(32*v)==0 && uintptr_t(x.data_ptr())%(sizeof(scalar_t)*v)==0 && uintptr_t(dy.data_ptr())%(sizeof(scalar_t)*v)==0 && (!w.defined()||uintptr_t(w.data_ptr())%(sizeof(A)*v)==0);};
      if(D==128&&aligned(4)){grid=(P+threads/8-1)/(threads/8);WIDTH(16,4,128);partials=grid;}
      else if(D==64&&aligned(2)){grid=(P+threads/8-1)/(threads/8);WIDTH(8,2,64);partials=grid;}
      else if(D==256&&aligned(4)&&2*(threads/32)*D*sizeof(A)<=48*1024){WIDTH(8,4,256);partials=grid;}
      else if(D==384&&aligned(4)&&2*(threads/32)*D*sizeof(A)<=48*1024){WIDTH(12,4,384);partials=grid;}
      else if(D==512&&aligned(4)&&2*(threads/32)*D*sizeof(A)<=48*1024){WIDTH(16,4,512);partials=grid;}
      else if(aligned(4)) {
        if(D==256){OLDWIDTH(8,4,256)}else if(D==384){OLDWIDTH(12,4,384)}else if(D==512){OLDWIDTH(16,4,512)}else if(D==768){OLDWIDTH(24,4,768)}else if(D==1024){OLDWIDTH(32,4,1024)}else{OLDWIDTH(32,4,0)}
      }else{
        if(D<=128){OLDWIDTH(4,1,0)}else if(D<=256){OLDWIDTH(8,1,0)}else if(D<=512){OLDWIDTH(16,1,0)}else{OLDWIDTH(32,1,0)}
      }
    }else if(M) {
      int64_t grid=(M+threads/32-1)/(threads/32);
      if(rms) backward_dx<scalar_t,A,true><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,mean.data_ptr<A>(),inv.data_ptr<A>(),dx.data_ptr<scalar_t>(),M,D);
      else backward_dx<scalar_t,A,false><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,mean.data_ptr<A>(),inv.data_ptr<A>(),dx.data_ptr<scalar_t>(),M,D);
    }
    if(w.defined()||has_bias) {
      if(M && D>1024) affine_partial<scalar_t,A><<<dim3((D+127)/128,P),128,0,stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),mean.data_ptr<A>(),inv.data_ptr<A>(),w.defined()?pw.data_ptr<A>():nullptr,has_bias?pb.data_ptr<A>():nullptr,M,D,rows);
      affine_finish<A><<<D,256,0,stream>>>(w.defined()?pw.data_ptr<A>():nullptr,has_bias?pb.data_ptr<A>():nullptr,w.defined()?dw.data_ptr<A>():nullptr,has_bias?db.data_ptr<A>():nullptr,partials,D);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dw,db};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&norm_forward);m.def("backward",&norm_backward);}
