// Experimental full-row gate reduction: no row atomics or zero-fill pass.
template<int D, int RR>
__device__ __forceinline__ void gate_row(const Gate& p) {
  using namespace wbwd;
  constexpr int G8 = D / 8, NI = (G8 + 31) / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float sp[NI][8], ep[NI][8], c0[NI][8], c1[NI][8];
#pragma unroll
  for (int it=0; it<NI; ++it)
#pragma unroll
    for (int j=0; j<8; ++j) {
      const int c=(lane+32*it)*8+j;
      sp[it][j] = c<D ? p.vec[c] : 0.f;
      ep[it][j] = c<D ? p.vec[D+c] : 0.f;
      c0[it][j]=0.f; c1[it][j]=0.f;
    }
  for (int blk=blockIdx.x; blk*64<p.M; blk+=gridDim.x)
  for (int rr=0; rr<8; rr+=RR) {
    const int row0=blk*64+warp*8+rr;
    uint4 vy[RR][NI], vp[RR][NI], vg[RR][NI];
#pragma unroll
    for (int r=0;r<RR;++r)
#pragma unroll
      for (int it=0;it<NI;++it) {
        const int q=lane+32*it, row=row0+r;
        if (q<G8 && row<p.M) {
          const size_t e=(size_t)row*G8+q;
          vy[r][it]=reinterpret_cast<const uint4*>(p.dy)[e];
          vp[r][it]=reinterpret_cast<const uint4*>(p.pv)[e];
          vg[r][it]=reinterpret_cast<const uint4*>(p.gv)[e];
        }
      }
#pragma unroll
    for (int r=0;r<RR;++r) {
      const int row=row0+r;
      if (row>=p.M) break;
      const float ro=p.rs[row], rm=ro*p.mu[row];
      float s1=0.f, s2=0.f;
#pragma unroll
      for (int it=0;it<NI;++it) {
        const int q=lane+32*it;
        if (q>=G8) break;
        float fy[8], fp[8], fg[8], fd[8], dpr[8], dg[8];
        unpack8(vy[r][it],fy); unpack8(vp[r][it],fp); unpack8(vg[r][it],fg);
        if (p.ds) unpack8(*reinterpret_cast<const uint4*>(p.ds+(size_t)(row%p.L)*D+8*q),fd);
        else {
#pragma unroll
          for (int j=0;j<8;++j) fd[j]=1.f;
        }
#pragma unroll
        for (int j=0;j<8;++j) {
          const float sg=sigmoidf_fast(fg[j]), dp=fy[j]*fd[j]*sg;
          dpr[j]=dp*ro; dg[j]=dp*fp[j]*(1.f-sg);
          s1=fmaf(dp,sp[it][j],s1); s2=fmaf(dp,fp[j]-ep[it][j],s2);
          c0[it][j]+=dp; c1[it][j]=fmaf(dp,rm,c1[it][j]);
        }
        reinterpret_cast<uint4*>(p.dpr)[(size_t)row*G8+q]=pack8(dpr);
        *reinterpret_cast<uint4*>(p.dg+(size_t)row*p.ldg+8*q)=pack8(dg);
      }
      s1=warp_sum(s1); s2=warp_sum(s2);
      if (lane==0) { p.s1[row]=s1; p.s2[row]=s2; }
    }
  }
  __shared__ float cs[2][8][D];
#pragma unroll
  for (int it=0;it<NI;++it) {
    const int q=lane+32*it;
    if (q>=G8) break;
#pragma unroll
    for (int j=0;j<8;++j) {
      cs[0][warp][q*8+j]=c0[it][j]; cs[1][warp][q*8+j]=c1[it][j];
    }
  }
  __syncthreads();
  for (int c=threadIdx.x;c<D;c+=256) {
    float a=0.f,b=0.f;
#pragma unroll
    for (int w=0;w<8;++w) { a+=cs[0][w][c]; b+=cs[1][w][c]; }
    p.part[(size_t)blockIdx.x*2*D+c]=a;
    p.part[(size_t)blockIdx.x*2*D+D+c]=b;
  }
}
