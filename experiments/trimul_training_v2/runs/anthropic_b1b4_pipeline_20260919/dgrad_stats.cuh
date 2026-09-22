// LN row reductions in the GEMM register layout; channel-contiguous epilogue.
TMN_DEVI void udgrad_stats(const Params& p,uint8_t* sm,uint64_t* bar,int m0){
 const int tid=threadIdx.x,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;
 uint8_t *sn=sm+147456,*sx=sm+98304;float* stats=reinterpret_cast<float*>(sm+32768);
 float* mus=stats+128,*rss=mus+64,*gam=rss+64;
 if(tid<64){mus[tid]=p.mean[m0+tid];rss[tid]=p.rs[m0+tid];}gam[tid]=p.gamma[tid];gam[tid+128]=p.gamma[tid+128];sync_group();
 int ra=w*16+lane/4,rb=ra+8;float mu[2]={mus[ra],mus[rb]},rs[2]={rss[ra],rss[rb]},s1[2]={},s2[2]={};
#pragma unroll
 for(int qn=0;qn<4;++qn){int n=(qn+3)%4;uint8_t* sw=sm+(n==0?131072:n==1?65536:n==2?81920:147456);
  float acc[32]={};fence_regs(acc);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_dgrad(acc,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);
#pragma unroll
  for(int q=0;q<4;++q){uint32_t fx[4],dn[4];ldsm_x4_t(fx,smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
#pragma unroll
   for(int j=0;j<4;++j){dn[j]=pack_bf16(acc[q*8+j*2],acc[q*8+j*2+1]);int rr=j&1,c=n*64+q*16+2*(lane%4)+8*(j>>1);float xa=__fmul_rn(__fsub_rn(bf16lo(fx[j]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(fx[j]),mu[rr]),rs[rr]);float ha=bf16lo(dn[j])*gam[c],hb=bf16hi(dn[j])*gam[c+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;}
   int chq=8*(mat>>1)+r8;stsm_x4_t(smem_u32(sn)+swz128(n*64+q*16+chq,(w*16+8*(mat&1))*2),dn[0],dn[1],dn[2],dn[3]);
  }
  sync_group();
 }
 s1[0]=quad_sum(s1[0])/256.f;s1[1]=quad_sum(s1[1])/256.f;s2[0]=quad_sum(s2[0])/256.f;s2[1]=quad_sum(s2[1])/256.f;
 if(lane%4==0){stats[ra*2]=s1[0];stats[ra*2+1]=s2[0];stats[rb*2]=s1[1];stats[rb*2+1]=s2[1];}sync_group();
 float* red=reinterpret_cast<float*>(sm+180224);
#pragma unroll 1
 for(int b=0;b<16;++b){int c=b*16+w*4+lane/8;float dg=0,db=0,gamma=gam[c];
#pragma unroll
  for(int k=0;k<4;++k){int r=2*(lane%8)+16*k;uint32_t x=pair_get(sx,c,r),dn=pair_get(sn,c,r);float mua=mus[r],mub=mus[r+1],rsa=rss[r],rsb=rss[r+1];float xa=__fmul_rn(__fsub_rn(bf16lo(x),mua),rsa),xb=__fmul_rn(__fsub_rn(bf16hi(x),mub),rsb),da=bf16lo(dn),dd=bf16hi(dn);dg+=da*xa+dd*xb;db+=da+dd;pair_put(sx,c,r,pack_bf16(rsa*((da*gamma-stats[r*2+1])-xa*stats[r*2]),rsb*((dd*gamma-stats[(r+1)*2+1])-xb*stats[(r+1)*2])));}
#pragma unroll
  for(int sh=1;sh<8;sh*=2){dg+=__shfl_xor_sync(0xffffffff,dg,sh);db+=__shfl_xor_sync(0xffffffff,db,sh);}
  if(lane%8==0){red[c]+=dg;red[256+c]+=db;}
 }
 sync_group();fence_proxy_async();sync_group();if(tid==0){for(int c=0;c<256;c+=16)tma_store_3d(&p.dtri,sx+c*128,m0,c,0);tma_store_commit();tma_store_wait_all();}sync_group();
}
