// SPDX-License-Identifier: Apache-2.0
// One computing WG owns all 128 output channels. Two CTAs can share an SM.
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma,const float* beta){
 int split=blockIdx.x-(SPLIT_ROLE==2?0:DWCOUNT),lane=threadIdx.x%32,w=(threadIdx.x%128)/32,tid=threadIdx.x%128;
 int ra=w*16+lane/4,rb=ra+8;float run_g=0,run_b=0;int round=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64;float acc[64]={};
  uint32_t ma=__bfloat16_as_ushort(p.mask[row+ra]),mb=__bfloat16_as_ushort(p.mask[row+rb]);ma|=ma<<16;mb|=mb<<16;
  
#if !DX_PC
  if(tid==0){mbar_arrive_expect_tx(bar+4,16384);for(int c=0;c<2;++c)tma_load_2d(sm+81920+c*8192,&p.xn,bar+4,c*64,row);}
#endif
  mbar_wait(bar+4,round&1);
  {
   uint32_t xn[8][4];load_frag_bf16<8,8192>(xn,smem_u32(sm+(NEXTROW?98304:81920)),w*16,lane);allsync();
   if(!DX_PC&&tid==0&&round>0)tma_store_wait_all();allsync();
   for(int side=0;side<2;++side){
    
#if !DX_PC
    load_g(p,sm,bar,row,side,0);load_g(p,sm,bar,row,side,1);
#endif

    for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,h/2);
     single_derivatives(p,xn,s,s+32768,s+32768,sm+81920+h*8192,row,ma,mb);
     fence_regs(acc);wgmma_fence();
     static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
      single_dx(acc,smem_desc(smem_u32(s+32768+k*2048),16,1024,1),smem_desc(smem_u32(s+(k/2)*16384+(k%2)*2048),8192,1024,1),side>0||h>0||k>0);
     });wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
#if DX_PC
     if(tid==0)mbar_arrive(bar+8+slot);
#else
     if(h<2)load_g(p,sm,bar,row,side,h+2);
#endif
    }
    
#if !DX_PC
    load_p(p,sm,bar,side,0);load_p(p,sm,bar,side,1);
#endif

    for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,h/2);
     fence_regs(acc);wgmma_fence();
     static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
      proj128(acc,smem_desc(smem_u32(sm+81920+h*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+k*32),16,1024,1),1);
     });wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
#if DX_PC
     if(tid==0)mbar_arrive(bar+8+slot);
#else
     if(h<2)load_p(p,sm,bar,side,h+2);
#endif
     
#if !DX_PC
    if(side==1&&h==1)load_ln_next(p,sm+65536,bar+3,row);
#endif

    }
   }
  }
  
#if !DX_PC
    issue_gate(p,sm,bar+2,row);
#endif
mbar_wait(bar+2,round&1);
  {
   float gate[64]={};fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;
    gate128(gate,smem_desc(smem_u32(sm+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+16384+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;acc[j]=math::round_bf16(acc[j]+math::round_bf16(gate[j]));});
  }
#if NEXTROW >= 2
  allsync();if(tid==0)mbar_arrive(bar+6);
#endif
  uint8_t* lnsm=sm+65536;mbar_wait(bar+3,round&1);
  uint32_t raw[8][4];load_frag_bf16<8,8192>(raw,smem_u32(lnsm),w*16,lane);
  LnStats st=ln_stats_only<8>(raw,gamma,beta,lane,1e-5f);
  float mu[2]={st.mA,st.mB},rs[2]={st.rA,st.rB},s1[2][2]={},s2[2][2]={};
  static_for<8>([&](auto qq){constexpr int q=decltype(qq)::value;
   static_for<4>([&](auto jj){constexpr int j=decltype(jj)::value;int rr=j&1,c=q*16+2*(lane%4)+8*(j>>1);
    float xa=__fmul_rn(__fsub_rn(bf16lo(raw[q][j]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(raw[q][j]),mu[rr]),rs[rr]);
    float ha=acc[q*8+j*2]*gamma[c],hb=acc[q*8+j*2+1]*gamma[c+1];s1[q/4][rr]+=ha*xa+hb*xb;s2[q/4][rr]+=ha+hb;
   });
  });
  float c1[2],c2[2];
  static_for<2>([&](auto ii){constexpr int i=decltype(ii)::value;c1[i]=quad_sum(s1[0][i])/128.f+quad_sum(s1[1][i])/128.f;c2[i]=quad_sum(s2[0][i])/128.f+quad_sum(s2[1][i])/128.f;});
  allsync(); // All LN source reads finish before in-place dx stores.
  float* tmp=reinterpret_cast<float*>(NEXTROW?sm+57344:lnsm+32768);
  static_for<8>([&](auto qq){constexpr int q=decltype(qq)::value;
   static_for<2>([&](auto pp){constexpr int pair=decltype(pp)::value;int j=pair*2,c=q*16+2*(lane%4)+8*pair;
    float xaa=__fmul_rn(__fsub_rn(bf16lo(raw[q][pair*2]),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(raw[q][pair*2]),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(bf16lo(raw[q][pair*2+1]),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(raw[q][pair*2+1]),mu[1]),rs[1]);
    float da=acc[q*8+pair*4],db=acc[q*8+pair*4+1],dc=acc[q*8+pair*4+2],dd=acc[q*8+pair*4+3],ga=gamma[c],gb=gamma[c+1];
    uint32_t oa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
    uint32_t ob=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
    uint8_t* out=lnsm+(q/4)*8192;uint8_t* res=lnsm+16384+(q/4)*8192;
    uint32_t resa=pair_get(res,ra,c%64),resb=pair_get(res,rb,c%64);
    *reinterpret_cast<uint32_t*>(out+swz128(ra,(c%64)*2))=pack_bf16(bf16lo(oa)+bf16lo(resa),bf16hi(oa)+bf16hi(resa));
    *reinterpret_cast<uint32_t*>(out+swz128(rb,(c%64)*2))=pack_bf16(bf16lo(ob)+bf16lo(resb),bf16hi(ob)+bf16hi(resb));
    float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
    for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
    if(lane<4){tmp[w*256+c]=dga;tmp[w*256+c+1]=dgb;tmp[w*256+128+c]=dba;tmp[w*256+128+c+1]=dbb;}
   });
  });allsync();
  run_g+=(tmp[tid]+tmp[256+tid])+(tmp[512+tid]+tmp[768+tid]);
  run_b+=(tmp[128+tid]+tmp[384+tid])+(tmp[640+tid]+tmp[896+tid]);
  fence_proxy_async();allsync();
  if(tid==0){store2d(&p.dx,lnsm,0,row);store2d(&p.dx,lnsm+8192,64,row);tma_store_commit();}
#if DX_PC
  if(tid==0)tma_store_wait_all();allsync();if(tid==0)mbar_arrive(bar+5);
#endif
 }
 if(tid==0)tma_store_wait_all();allsync();p.partln[split*256+tid]=run_g;p.partln[split*256+128+tid]=run_b;
}
