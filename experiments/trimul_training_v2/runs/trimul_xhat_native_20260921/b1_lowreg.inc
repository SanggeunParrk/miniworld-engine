// Materialize only the 64x256 BF16 dNorm tile in shared memory. It never
// reaches HBM. This bounds register lifetimes while dWproj accumulates.
TMN_DEVI void lowreg_dgrad(const Params& p,uint8_t* sm,int m0,int slot){
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;
 uint8_t* sx=sm+32768+slot*49152;
 uint8_t* sn=sm+16384+(1-slot)*49152;uint8_t* dp=XHAT_FP32==1?sn+32768:sm;
 float* stats=reinterpret_cast<float*>(sm+224256);
 float* mus=reinterpret_cast<float*>(sm+225280);float* rss=mus+64;float* gam=reinterpret_cast<float*>(sm+214016);
 int ra=w*16+lane/4,rb=ra+8;float rs[2]={rss[ra],rss[rb]},s1[2]={},s2[2]={};
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;uint8_t* sw=sm+147456+n*16384;float acc[32]={};
  fence_regs(acc);wgmma_fence();static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(acc,smem_desc(smem_u32(dp+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc);
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;uint32_t dn[4];float h[8];xhat_frag(h,sm,slot,w,lane,n*4+q);
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;dn[j]=pack_bf16(acc[q*8+j*2],acc[q*8+j*2+1]);int rr=j&1,c=n*64+q*16+2*(lane%4)+8*(j>>1);
    float xa=h[j*2],xb=h[j*2+1];
    float ha=bf16lo(dn[j])*gam[c],hb=bf16hi(dn[j])*gam[c+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;
   });
   stsm_x4(smem_u32(sn+n*8192)+swz128(w*16+r8+8*(mat&1),(2*q+(mat>>1))*16),dn[0],dn[1],dn[2],dn[3]);
  });
 });
 s1[0]=quad_sum(s1[0])/256.f;s1[1]=quad_sum(s1[1])/256.f;s2[0]=quad_sum(s2[0])/256.f;s2[1]=quad_sum(s2[1])/256.f;
 if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}allsync();
 float c1a=stats[ra*2]+stats[128+ra*2],c1b=stats[rb*2]+stats[128+rb*2],c2a=stats[ra*2+1]+stats[128+ra*2+1],c2b=stats[rb*2+1]+stats[128+rb*2+1];
 float* tmp=reinterpret_cast<float*>(sm+216064);
 uint32_t out_cache[32];
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;uint32_t dn[4],out[4];float h[8];xhat_frag(h,sm,slot,w,lane,n*4+q);
   ldsm_x4(dn,smem_u32(sn+n*8192)+swz128(w*16+r8+8*(mat&1),(2*q+(mat>>1))*16));
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;constexpr int j=pair*2;int c=n*64+q*16+2*(lane%4)+8*pair;
    float xaa=h[j*2],xab=h[j*2+1];
    float xba=h[j*2+2],xbb=h[j*2+3];
    float daa=bf16lo(dn[j]),dab=bf16hi(dn[j]),dba=bf16lo(dn[j+1]),dbb=bf16hi(dn[j+1]),ga=gam[c],gb=gam[c+1];
    out[j]=pack_bf16(rs[0]*((daa*ga-c2a)-xaa*c1a),rs[0]*((dab*gb-c2a)-xab*c1a));out[j+1]=pack_bf16(rs[1]*((dba*ga-c2b)-xba*c1b),rs[1]*((dbb*gb-c2b)-xbb*c1b));
    float dga=daa*xaa+dba*xba,dgb=dab*xab+dbb*xbb,dba0=daa+dba,dbb0=dab+dbb;
    #pragma unroll
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba0+=__shfl_xor_sync(0xffffffff,dba0,sh);dbb0+=__shfl_xor_sync(0xffffffff,dbb0,sh);}
    if(lane<4){tmp[w*512+c]=dga;tmp[w*512+c+1]=dgb;tmp[w*512+256+c]=dba0;tmp[w*512+256+c+1]=dbb0;}
   });
   // Each lane reads and writes the same raw-tri elements; no inter-lane
   // dependence exists between raw loads and transposed derivative stores.
#if XHAT_FP32 == 1
   #pragma unroll
   for(int j=0;j<4;++j)out_cache[nl*16+q*4+j]=out[j];
#else
   stsm_x4_t(smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2),out[0],out[1],out[2],out[3]);
#endif
  });
 });allsync();
#if XHAT_FP32 == 1
 // All dNorm reads have completed: recycle its 32 KiB region for dTri.
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;int k=nl*16+q*4;
   stsm_x4_t(smem_u32(sn)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2),out_cache[k],out_cache[k+1],out_cache[k+2],out_cache[k+3]);
  });
 });allsync();
 fence_proxy_async();sync_group();if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=16)tma_store_3d(&p.dtri,sn+ch*128,m0,ch,0);tma_store_commit();tma_store_wait_all();}sync_group();
#endif
 int c=threadIdx.x;float* red=reinterpret_cast<float*>(sm+225792);red[c]+=(tmp[c]+tmp[512+c])+(tmp[1024+c]+tmp[1536+c]);red[256+c]+=(tmp[256+c]+tmp[768+c])+(tmp[1280+c]+tmp[1792+c]);
 fence_proxy_async();sync_group();if(XHAT_FP32!=1 && tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=16)tma_store_3d(&p.dtri,
#if XHAT_FP32 == 1
 (wi==0?sm+16384+slot*49152:sm)+(ch-wi*128)*128,
#else
 sx+ch*128,
#endif
 m0,ch,0);tma_store_commit();tma_store_wait_all();}sync_group();
}
