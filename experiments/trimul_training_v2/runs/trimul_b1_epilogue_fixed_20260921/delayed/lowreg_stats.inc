// 32 KiB of dead dy/current x_n storage. Row swizzle avoids bank conflicts
// when 8 lanes per column publish two-row partials.
TMN_DEVI uint32_t param_ptr(uint8_t* sm,int slot,int r,int c){
 return smem_u32(r<16?sm:sm+16384+slot*49152)+(r%16)*1024+(c^((r&3)*8))*4;
}
TMN_DEVI void param_store(uint8_t* sm,int slot,int r,int c,float a,float b){
 asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(param_ptr(sm,slot,r,c)),"f"(a),"f"(b) : "memory");
}
TMN_DEVI float param_reduce(uint8_t* sm,int slot,int c){
 float parts[4];
 constexpr int param_unroll=B1_PARAM_UNROLL;
 #pragma unroll param_unroll
 for(int w=0;w<4;++w){float v[8];
  static_for<8>([&](auto ri){constexpr int r=decltype(ri)::value;asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v[r]) : "r"(param_ptr(sm,slot,w*8+r,c)) : "memory");});
  parts[w]=__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),__fadd_rn(v[2],v[3])),__fadd_rn(__fadd_rn(v[4],v[5]),__fadd_rn(v[6],v[7])));
 }
 return __fadd_rn(__fadd_rn(parts[0],parts[1]),__fadd_rn(parts[2],parts[3]));
}
// Selected path keeps BF16 dNorm fragments in registers. The alternative
// shared dNorm path is retained for ablations. No dNorm is written to HBM.
TMN_DEVI void lowreg_dgrad(const Params& p,uint8_t* sm,uint64_t* bars,int m0,int slot){
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;
 uint8_t* sx=sm+32768+slot*49152;
 uint8_t* sn=sm+16384+(1-slot)*49152;
 uint8_t* dp=B1_DIRECT_DP?sn+32768:sm;
 uint32_t dnreg[2][4][4];
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
 #if B1_REG_EARLY_RAW
 // wait_group releases operands only for this warp-group. Both groups
 // read dProj, so thread 0 must not overwrite it until group 1 also finishes.
 allsync();
 // dProj is no longer consumed and dNorm stays in registers. Both stats
 // slots are independent; current tri stays untouched by the next input load.
 if(m0/64+UCOUNT<p.tiles)load_raw(p,sm,bars,1-slot,m0+UCOUNT*64);
 #endif

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
#if B1_DN_REG
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;dnreg[nl][q][j]=dn[j];});
#else
   stsm_x4(smem_u32((B1_SPLIT_DN ? (n<2?sm:sm+16384+slot*49152)+(n%2)*8192 : sn+n*8192))+swz128(w*16+r8+8*(mat&1),(2*q+(mat>>1))*16),dn[0],dn[1],dn[2],dn[3]);
#endif
  });
 });
 s1[0]=quad_sum(s1[0])/256.f;s1[1]=quad_sum(s1[1])/256.f;s2[0]=quad_sum(s2[0])/256.f;s2[1]=quad_sum(s2[1])/256.f;
 if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}allsync();
 #if B1_SPLIT_DN && !B1_REG_EARLY_RAW
 // All WGMMA dNorm operands are released. The opposite 48 KiB slot
 // can prefetch the complete next input.
 // Stats use a distinct slot too: current mu/rstd cannot be overwritten.
 if(m0/64+UCOUNT<p.tiles)load_raw(p,sm,bars,1-slot,m0+UCOUNT*64);
#endif
 float c1a=stats[ra*2]+stats[128+ra*2],c1b=stats[rb*2]+stats[128+rb*2],c2a=stats[ra*2+1]+stats[128+ra*2+1],c2b=stats[rb*2+1]+stats[128+rb*2+1];
 float* tmp=reinterpret_cast<float*>(sm+216064);
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;uint32_t fx[4],dn[4],out[4];
   ldsm_x4_t(fx,smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
#if B1_DN_REG
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;dn[j]=dnreg[nl][q][j];});
#else
   ldsm_x4(dn,smem_u32((B1_SPLIT_DN ? (n<2?sm:sm+16384+slot*49152)+(n%2)*8192 : sn+n*8192))+swz128(w*16+r8+8*(mat&1),(2*q+(mat>>1))*16));
#endif
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;constexpr int j=pair*2;int c=n*64+q*16+2*(lane%4)+8*pair;
    float xaa=__fmul_rn(__fsub_rn(bf16lo(fx[j]),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(fx[j]),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(bf16lo(fx[j+1]),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(fx[j+1]),mu[1]),rs[1]);
    float daa=bf16lo(dn[j]),dab=bf16hi(dn[j]),dba=bf16lo(dn[j+1]),dbb=bf16hi(dn[j+1]),ga=gam[c],gb=gam[c+1];
    out[j]=pack_bf16(rs[0]*((daa*ga-c2a)-xaa*c1a),rs[0]*((dab*gb-c2a)-xab*c1a));out[j+1]=pack_bf16(rs[1]*((dba*ga-c2b)-xba*c1b),rs[1]*((dbb*gb-c2b)-xbb*c1b));
    float dga=daa*xaa+dba*xba,dgb=dab*xab+dbb*xbb,dba0=daa+dba,dbb0=dab+dbb;
#if B1_PARAM_SHARED
    param_store(sm,slot,w*8+lane/4,c,dga,dgb);
#else
    #pragma unroll
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba0+=__shfl_xor_sync(0xffffffff,dba0,sh);dbb0+=__shfl_xor_sync(0xffffffff,dbb0,sh);}
    if(lane<4){tmp[w*512+c]=dga;tmp[w*512+c+1]=dgb;tmp[w*512+256+c]=dba0;tmp[w*512+256+c+1]=dbb0;}
#endif
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
#if B1_DTRI_ASYNC_EPI
 fence_proxy_async();sync_group();if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=B1_DTRI_STORE_C)tma_store_3d(&p.dtri,sx+ch*128,m0,ch,0);tma_store_commit();}
#endif
 int c=threadIdx.x;float* red=reinterpret_cast<float*>(sm+225792);
#if B1_PARAM_SHARED
 red[c]+=param_reduce(sm,slot,c);allsync();
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;constexpr int j=pair*2;int col=n*64+q*16+2*(lane%4)+8*pair;
    float a=bf16lo(dnreg[nl][q][j])+bf16lo(dnreg[nl][q][j+1]);
    float b=bf16hi(dnreg[nl][q][j])+bf16hi(dnreg[nl][q][j+1]);
    param_store(sm,slot,w*8+lane/4,col,a,b);
   });
  });
 });allsync();red[256+c]+=param_reduce(sm,slot,c);allsync();
#else
 red[c]+=(tmp[c]+tmp[512+c])+(tmp[1024+c]+tmp[1536+c]);red[256+c]+=(tmp[256+c]+tmp[768+c])+(tmp[1280+c]+tmp[1792+c]);
#endif
#if B1_DTRI_ASYNC_EPI
 if(tid==0)tma_store_wait_all();sync_group();
#else
 fence_proxy_async();sync_group();if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=B1_DTRI_STORE_C)tma_store_3d(&p.dtri,sx+ch*128,m0,ch,0);tma_store_commit();tma_store_wait_all();}sync_group();
#endif
}
