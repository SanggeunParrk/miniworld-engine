struct MaskCycle{uint32_t b0,b1,b2,scale;int period;bool cached;};
template<int STRIDE> TMN_DEVI MaskCycle mask_cycle(const Params& p,int first){
 MaskCycle v={0,0,0,0,0,false};int a=64*STRIDE,b=p.L;while(b){int r=a%b;a=b;b=r;}
 v.period=p.L/a;v.cached=v.period<=3;
 if(v.cached&&first<p.tiles)for(int z=0;z<v.period;++z){uint32_t bits=0;int j0=((first+z*STRIDE)*64)%p.L;
  for(int i=threadIdx.x;i<1024;i+=256){int cb=i/512,r=(i%512)/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
   uint4 ds=ldg128(p.ds+jr*128+cb*64+c);uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);
#pragma unroll
   for(int q=0;q<4;++q){uint32_t lo=dd[q]&65535u,hi=dd[q]>>16;int bit=8*(i/256)+2*q;
    bits|=(uint32_t(lo!=0)<<bit)|(uint32_t(hi!=0)<<(bit+1));v.scale|=lo|hi;}
  }
  if(z==0)v.b0=bits;else if(z==1)v.b1=bits;else v.b2=bits;
 }
 return v;
}
template<bool DW> TMN_DEVI void gate_backward(const Params& p,uint8_t* s,int row,const MaskCycle& mask,int mi){
 int j0=row%p.L;uint32_t bits=mi==0?mask.b0:mi==1?mask.b1:mask.b2;
 for(int i=threadIdx.x;i<1024;i+=256){int cb=i/512,r=(i%512)/8,c=(i%8)*8,jr=j0+r;if(jr>=p.L)jr-=p.L;
  uint32_t sy=smem_u32(s+cb*8192)+swz128(r,c*2),sg=smem_u32(s+16384+cb*8192)+swz128(r,c*2);
  uint4 y=lds128(sy),g=lds128(sg),v,ds,dp,dg;
  if constexpr(DW)v=lds128(smem_u32(s+32768+cb*8192)+swz128(r,c*2));
  if(mask.cached){uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);
#pragma unroll
   for(int q=0;q<4;++q){int bit=8*(i/256)+2*q;dd[q]=((bits>>bit)&1?mask.scale:0)|((bits>>(bit+1))&1?mask.scale<<16:0);}
  }else ds=ldg128(p.ds+jr*128+cb*64+c);
  uint32_t *yy=reinterpret_cast<uint32_t*>(&y),*gg=reinterpret_cast<uint32_t*>(&g),*vv=reinterpret_cast<uint32_t*>(&v),*dd=reinterpret_cast<uint32_t*>(&ds),*oo=reinterpret_cast<uint32_t*>(&dp),*zz=reinterpret_cast<uint32_t*>(&dg);
#pragma unroll
  for(int q=0;q<4;++q){float ya=bf16lo(yy[q])*bf16lo(dd[q]),yb=bf16hi(yy[q])*bf16hi(dd[q]),ga=bf16lo(gg[q]),gb=bf16hi(gg[q]);oo[q]=pack_bf16(ya*ga,yb*gb);
   if constexpr(DW)zz[q]=pack_bf16(((ya*bf16lo(vv[q]))*ga)*(1.f-ga),((yb*bf16hi(vv[q]))*gb)*(1.f-gb));
  }
  asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(sy),"r"(dp.x),"r"(dp.y),"r"(dp.z),"r"(dp.w):"memory");
  if constexpr(DW){asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(sg),"r"(dg.x),"r"(dg.y),"r"(dg.z),"r"(dg.w):"memory");stg128(p.dg+(size_t)(row+r)*128+cb*64+c,dg);}
 }
 // Publish generic B1 stores to the WGMMA proxy, then join CTA threads.
 fence_proxy_async();allsync();
}
// Anthropic WGMMA/ldmatrix/stmatrix fragment conventions, applied to LN backward.
// This is only for the separate DX CTA: no 192-register dW live set is present.
TMN_DEVI void dual_dgrad(const Params& p,uint8_t* sm,int m0,int slot){
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;
 uint8_t* sx=sm+slot*98304+32768;
 float* stats=reinterpret_cast<float*>(sm+slot*98304+16384);
 // 73728..74752: double-buffered saved mean/rstd. Gamma is resident
 // at74752..75776. B1's existing CTA barrier publishes the row statistics.
 float* mus=reinterpret_cast<float*>(sm+73728+slot*512);
 float* rss=mus+64;float* gam=reinterpret_cast<float*>(sm+74752);
 int ra=w*16+lane/4,rb=ra+8;
 float mu[2]={mus[ra],mus[rb]},rs[2]={rss[ra],rss[rb]},s1[2]={},s2[2]={};
 uint32_t fx[2][4][4],dn[2][4][4];
 static_for<2>([&](auto ni){constexpr int nlocal=decltype(ni)::value;int n=wi*2+nlocal;
  uint8_t* sw=sm+163840+n*16384;float acc[32]={};
  fence_regs(acc);wgmma_fence(); // Order initialized accumulators before async MMA.
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(acc,smem_desc(smem_u32(sm+slot*98304+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc); // dnorm ready for BF16 rounding.
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   ldsm_x4_t(fx[nlocal][q],smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;
    dn[nlocal][q][j]=pack_bf16(acc[q*8+j*2],acc[q*8+j*2+1]);
    int rr=j&1,c=n*64+q*16+2*(lane%4)+8*(j>>1);
    float xa=__fmul_rn(__fsub_rn(bf16lo(fx[nlocal][q][j]),mu[rr]),rs[rr]);
    float xb=__fmul_rn(__fsub_rn(bf16hi(fx[nlocal][q][j]),mu[rr]),rs[rr]);
    float ha=bf16lo(dn[nlocal][q][j])*gam[c],hb=bf16hi(dn[nlocal][q][j])*gam[c+1];
    s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;
   });
  });
 });
 s1[0]=quad_sum(s1[0])/256.f;s1[1]=quad_sum(s1[1])/256.f;
 s2[0]=quad_sum(s2[0])/256.f;s2[1]=quad_sum(s2[1])/256.f;
 if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}
 allsync(); // Both channel halves of each row must be published before LN epilogue.
 float c1a=stats[ra*2]+stats[128+ra*2],c1b=stats[rb*2]+stats[128+rb*2];
 float c2a=stats[ra*2+1]+stats[128+ra*2+1],c2b=stats[rb*2+1]+stats[128+rb*2+1];
 // 65536..73728 holds8KiB per-warp LN partials; no dnorm shared round trip.
 float* tmp=reinterpret_cast<float*>(sm+65536);
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;uint32_t out[4];
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;constexpr int j=pair*2;
    int c=n*64+q*16+2*(lane%4)+8*pair;
    uint32_t xa=fx[nl][q][j],xb=fx[nl][q][j+1],da=dn[nl][q][j],db=dn[nl][q][j+1];
    float xaa=__fmul_rn(__fsub_rn(bf16lo(xa),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(xa),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(bf16lo(xb),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(xb),mu[1]),rs[1]);
    float daa=bf16lo(da),dab=bf16hi(da),dba=bf16lo(db),dbb=bf16hi(db);
    float ga=gam[c],gb=gam[c+1];
    out[j]=pack_bf16(rs[0]*((daa*ga-c2a)-xaa*c1a),rs[0]*((dab*gb-c2a)-xab*c1a));
    out[j+1]=pack_bf16(rs[1]*((dba*ga-c2b)-xba*c1b),rs[1]*((dbb*gb-c2b)-xbb*c1b));
    float dga=daa*xaa+dba*xba,dgb=dab*xab+dbb*xbb,dba0=daa+dba,dbb0=dab+dbb;
    // Same columns, eight row groups in the warp (each contributes two rows).
#pragma unroll
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba0+=__shfl_xor_sync(0xffffffff,dba0,sh);dbb0+=__shfl_xor_sync(0xffffffff,dbb0,sh);}
    if(lane<4){tmp[w*512+c]=dga;tmp[w*512+c+1]=dgb;tmp[w*512+256+c]=dba0;tmp[w*512+256+c+1]=dbb0;}
   });
   // All raw tri values were loaded before this point. Store dtri in the same
   // transposed matrix layout, without materializing or rereading dnorm.
   int chq=8*(mat>>1)+r8;
   stsm_x4_t(smem_u32(sx)+swz128(n*64+q*16+chq,(w*16+8*(mat&1))*2),out[0],out[1],out[2],out[3]);
  });
 });
 allsync(); // Publish all four warps' parameter partials (and all stmatrix writes).
 int c=threadIdx.x;float* red=reinterpret_cast<float*>(sm+229376);
 red[c]+=(tmp[c]+tmp[512+c])+(tmp[1024+c]+tmp[1536+c]);
 red[256+c]+=(tmp[256+c]+tmp[768+c])+(tmp[1280+c]+tmp[1792+c]);
 fence_proxy_async();sync_group(); // Generic dtri stores visible before each WG's TMA read.
 if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=16)tma_store_3d(&p.dtri,sx+ch*128,m0,ch,0);tma_store_commit();tma_store_wait_all();}
 sync_group(); // TMA finished consuming this slot before cluster reuse.
}

