// Wide dgrad: four outstanding 64-column GEMMs and LN in accumulator layout.
// BF16 dnorm rounding remains explicit; no shared-memory dnorm materialization.
TMN_DEVI void dgrad_wide(const Params& p,uint8_t* sm,uint64_t* bar,int* last){
 const int tid=threadIdx.x%128,lane=tid%32,w=tid/32,m0=blockIdx.x*64;
 uint8_t *sy=sm,*sg=sm+8192,*sp=sm+16384,*sw=sm+32768,*sx=sm;
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}sync_group();
 uint32_t a[8][4];int j0=m0%p.L;
#pragma unroll
 for(int kc=0;kc<2;++kc){
  if(tid==0){mbar_arrive_expect_tx(bar,24576+(kc==0?65536:0));tma_load_2d(sy,&p.dy,bar,kc*64,m0);tma_load_2d(sg,&p.gate,bar,kc*64,m0);tma_load_2d(sp,&p.proj,bar,kc*64,m0);
   if(kc==0){for(int n=0;n<4;++n)for(int k=0;k<2;++k)tma_load_2d(sw+(n*2+k)*8192,&p.wp,bar,k*64,n*64);}
  }
  sync_group();mbar_wait(bar,kc&1);
  for(int i=tid;i<2048;i+=128){int r=i/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;uint32_t y=pair_get(sy,r,c),g=pair_get(sg,r,c),v=pair_get(sp,r,c),ds=ldg32(p.ds+jr*128+kc*64+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
   stg32(p.dg+(size_t)(m0+r)*128+kc*64+c,pack_bf16(((ya*bf16lo(v))*ga)*(1.f-ga),((yb*bf16hi(v))*gb)*(1.f-gb)));pair_put(sy,r,c,pack_bf16(ya*ga,yb*gb));
  }
  sync_group();uint32_t f[4][4];load_frag_bf16<4,8192>(f,smem_u32(sy),w*16,lane);
#pragma unroll
  for(int k=0;k<4;++k)for(int q=0;q<4;++q)a[kc*4+k][q]=f[k][q];
  fence_proxy_async();sync_group();
 }
 // All raw input readers finished. TMA of tri overlaps the four MMA groups.
 if(tid==0){mbar_arrive_expect_tx(bar,32768);tma_load_2d(sx,&p.tri,bar,m0,0);}
 float acc[4][32];
 static_for<4>([&](auto ni){constexpr int n=decltype(ni)::value;
  for(int j=0;j<32;++j)acc[n][j]=0;
  uint64_t de=smem_desc(smem_u32(sw+n*16384),16,1024,1);uint32_t lo[1]={(uint32_t)de},hi[1]={(uint32_t)(de>>32)};
  fence_regs(acc[n]);wgmma_fence();mma_chain<2,8,1>(acc[n],a,lo,hi);wgmma_commit();
 });
 wgmma_wait<0>();static_for<4>([&](auto ni){fence_regs(acc[decltype(ni)::value]);});
 mbar_wait(bar,0);sync_group();
 float* red=reinterpret_cast<float*>(sw);float s1[2]={},s2[2]={};
 int ra=m0+w*16+lane/4,rb=ra+8;float mu[2]={p.mean[ra],p.mean[rb]},rs[2]={p.rs[ra],p.rs[rb]};
 int mat=lane/8,r8=lane%8;
 static_for<16>([&](auto ki){constexpr int k=decltype(ki)::value;uint32_t fx[4];
  ldsm_x4_t(fx,smem_u32(sx)+swz128(k*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
  float gd[4]={},bd[4]={};
#pragma unroll
  for(int q=0;q<4;++q){int rr=q&1,cc=k*16+2*(lane%4)+8*(q>>1);float da=__bfloat162float(__float2bfloat16_rn(acc[k/4][(k%4)*8+q*2])),db=__bfloat162float(__float2bfloat16_rn(acc[k/4][(k%4)*8+q*2+1]));
   acc[k/4][(k%4)*8+q*2]=da;acc[k/4][(k%4)*8+q*2+1]=db;
   float xa=__fmul_rn(__fsub_rn(bf16lo(fx[q]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(fx[q]),mu[rr]),rs[rr]);float ha=da*p.gamma[cc],hb=db*p.gamma[cc+1];s1[rr]+=ha*xa+hb*xb;s2[rr]+=ha+hb;gd[2*(q>>1)]+=da*xa;gd[2*(q>>1)+1]+=db*xb;bd[2*(q>>1)]+=da;bd[2*(q>>1)+1]+=db;
  }
#pragma unroll
  for(int q=0;q<4;++q){
#pragma unroll
   for(int sh=4;sh<=16;sh*=2){gd[q]+=__shfl_xor_sync(0xffffffff,gd[q],sh);bd[q]+=__shfl_xor_sync(0xffffffff,bd[q],sh);}
   if(lane<4){int c=k*16+2*lane+8*(q/2)+q%2;red[w*512+c]=gd[q];red[w*512+256+c]=bd[q];}
  }
 });
 s1[0]=quad_sum(s1[0])/256.f;s1[1]=quad_sum(s1[1])/256.f;s2[0]=quad_sum(s2[0])/256.f;s2[1]=quad_sum(s2[1])/256.f;
 static_for<16>([&](auto ki){constexpr int k=decltype(ki)::value;uint32_t fx[4],fo[4];
  ldsm_x4_t(fx,smem_u32(sx)+swz128(k*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
#pragma unroll
  for(int q=0;q<4;++q){int rr=q&1,cc=k*16+2*(lane%4)+8*(q>>1);float xa=__fmul_rn(__fsub_rn(bf16lo(fx[q]),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(fx[q]),mu[rr]),rs[rr]);fo[q]=pack_bf16(rs[rr]*((acc[k/4][(k%4)*8+q*2]*p.gamma[cc]-s2[rr])-xa*s1[rr]),rs[rr]*((acc[k/4][(k%4)*8+q*2+1]*p.gamma[cc+1]-s2[rr])-xb*s1[rr]));}
  int chq=8*(mat>>1)+r8;uint32_t off=chq*128+(((2*w+(mat&1))^(chq&7))*16)+k*16*128;stsm_x4_t(smem_u32(sx)+off,fo[0],fo[1],fo[2],fo[3]);
 });
 sync_group();fence_proxy_async();sync_group();
 if(tid==0){for(int c=0;c<256;c+=16)tma_store_3d(&p.dtri,sx+c*128,m0,c,0);tma_store_commit();tma_store_wait_all();}
 sync_group();ln_finish(p,red,last);
}
