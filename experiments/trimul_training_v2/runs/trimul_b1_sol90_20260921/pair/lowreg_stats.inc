// Materialize only the 64x256 BF16 dNorm tile in shared memory. It never
// reaches HBM. This bounds register lifetimes while dWproj accumulates.
TMN_DEVI void lowreg_dgrad(const Params& p,uint8_t* sm,uint64_t* bars,int m0,int slot){
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;
 uint8_t* sx=sm+32768+slot*49152;
 uint8_t* sn=sm+16384+(1-slot)*49152;
 uint8_t* dp=B1_DIRECT_DP?sn+32768:sm;
 float* stats=reinterpret_cast<float*>(sm+224256);
 float* mus=reinterpret_cast<float*>(sm+B1_STATS_BASE(slot));float* rss=mus+64;float* gam=reinterpret_cast<float*>(sm+214016);
 int ra=w*16+lane/4,rb=ra+8;float mu[2]={mus[ra],mus[rb]},rs[2]={rss[ra],rss[rb]},s1[2]={},s2[2]={};
 #if B1_DN_PAIR
 float pair_acc[2][32]={};fence_regs(pair_acc[0]);fence_regs(pair_acc[1]);wgmma_fence();
 static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
  static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;uint8_t* sw=sm+147456+n*16384;
   mma_dgrad(pair_acc[nl],smem_desc(smem_u32(dp+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });
 });wgmma_commit();wgmma_wait<0>();fence_regs(pair_acc[0]);fence_regs(pair_acc[1]);
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;auto& acc=pair_acc[nl];
 #else
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;uint8_t* sw=sm+147456+n*16384;float acc[32]={};
  fence_regs(acc);wgmma_fence();static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(acc,smem_desc(smem_u32(dp+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc);
 #endif
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;uint32_t fx[4],dn[4];
   ldsm_x4_t(fx,smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;dn[j]=pack_bf16(acc[q*8+j*2],acc[q*8+j*2+1]);int rr=j&1,c=n*64+q*16+2*(lane%4)+8*(j>>1);
    float xa=__fmul_rn(__fsub_rn(bf16lo(fx[j]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(fx[j]),mu[rr]),rs[rr]);
    float ha=bf16lo(dn[j])*gam[c],hb=bf16hi(dn[j])*gam[c+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;
   });
   stsm_x4(smem_u32((B1_SPLIT_DN ? (n<2?sm:sm+16384+slot*49152)+(n%2)*8192 : sn+n*8192))+swz128(w*16+r8+8*(mat&1),(2*q+(mat>>1))*16),dn[0],dn[1],dn[2],dn[3]);
  });
 });
 s1[0]=quad_sum(s1[0])/256.f;s1[1]=quad_sum(s1[1])/256.f;s2[0]=quad_sum(s2[0])/256.f;s2[1]=quad_sum(s2[1])/256.f;
 if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}allsync();
 #if B1_SPLIT_DN
 // All WGMMA dNorm operands are released. dNorm lives in dead dy/x_n
 // storage, so the opposite 48 KiB slot can prefetch the complete next input.
 // Stats use a distinct slot too: current mu/rstd cannot be overwritten.
 if(m0/64+UCOUNT<p.tiles)load_raw(p,sm,bars,1-slot,m0+UCOUNT*64);
#endif
 float c1a=stats[ra*2]+stats[128+ra*2],c1b=stats[rb*2]+stats[128+rb*2],c2a=stats[ra*2+1]+stats[128+ra*2+1],c2b=stats[rb*2+1]+stats[128+rb*2+1];
 float* tmp=reinterpret_cast<float*>(sm+216064);
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;uint32_t fx[4],dn[4],out[4];
   ldsm_x4_t(fx,smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
   ldsm_x4(dn,smem_u32((B1_SPLIT_DN ? (n<2?sm:sm+16384+slot*49152)+(n%2)*8192 : sn+n*8192))+swz128(w*16+r8+8*(mat&1),(2*q+(mat>>1))*16));
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;constexpr int j=pair*2;int c=n*64+q*16+2*(lane%4)+8*pair;
    float xaa=__fmul_rn(__fsub_rn(bf16lo(fx[j]),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(fx[j]),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(bf16lo(fx[j+1]),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(fx[j+1]),mu[1]),rs[1]);
    float daa=bf16lo(dn[j]),dab=bf16hi(dn[j]),dba=bf16lo(dn[j+1]),dbb=bf16hi(dn[j+1]),ga=gam[c],gb=gam[c+1];
    out[j]=pack_bf16(rs[0]*((daa*ga-c2a)-xaa*c1a),rs[0]*((dab*gb-c2a)-xab*c1a));out[j+1]=pack_bf16(rs[1]*((dba*ga-c2b)-xba*c1b),rs[1]*((dbb*gb-c2b)-xbb*c1b));
    float dga=daa*xaa+dba*xba,dgb=dab*xab+dbb*xbb,dba0=daa+dba,dbb0=dab+dbb;
    #pragma unroll
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba0+=__shfl_xor_sync(0xffffffff,dba0,sh);dbb0+=__shfl_xor_sync(0xffffffff,dbb0,sh);}
    if(lane<4){tmp[w*512+c]=dga;tmp[w*512+c+1]=dgb;tmp[w*512+256+c]=dba0;tmp[w*512+256+c+1]=dbb0;}
   });
   // Each lane reads and writes the same raw-tri elements; no inter-lane
   // dependence exists between raw loads and transposed derivative stores.
   stsm_x4_t(smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2),out[0],out[1],out[2],out[3]);
  });
 });allsync();
#if B1_EARLY_RAW && !B1_SPLIT_DN
 // dNorm/dProj consumers have finished. The opposite slot can receive
 // the next tri/x_n/stats while the current dTri TMA store completes.
 if(m0/64+UCOUNT<p.tiles)load_raw(p,sm,bars,1-slot,m0+UCOUNT*64);
#endif
 int c=threadIdx.x;float* red=reinterpret_cast<float*>(sm+225792);red[c]+=(tmp[c]+tmp[512+c])+(tmp[1024+c]+tmp[1536+c]);red[256+c]+=(tmp[256+c]+tmp[768+c])+(tmp[1280+c]+tmp[1792+c]);
 fence_proxy_async();sync_group();if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=B1_DTRI_STORE_C)tma_store_3d(&p.dtri,sx+ch*128,m0,ch,0);tma_store_commit();tma_store_wait_all();}sync_group();
}
