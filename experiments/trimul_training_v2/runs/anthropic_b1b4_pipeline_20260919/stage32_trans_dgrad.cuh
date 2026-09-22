// Transpose dnorm GEMM: Wp.T[64,128] @ dp.T[128,32] -> [64,32].
// m64n32 consumes exactly32 real token rows; no dummy-row compute.
TMN_DEVI void stage_dgrad(const Params& p,uint8_t* sm,int slot,int m0){
 const int wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 uint8_t* s=sm+slot*65536;uint8_t *sn=sm+196608,*sx=s+49152;
 float* stats=reinterpret_cast<float*>(s+16384);
 float* mus=stats+1024,*rss=mus+32,*gam=rss+32;
 if(threadIdx.x<32){mus[tid]=p.mean[m0+tid];rss[tid]=p.rs[m0+tid];}gam[threadIdx.x]=p.gamma[threadIdx.x];
 // Publish saved metadata; dW and B1 finished before scratch proj is reused.
 allsync();
#pragma unroll
 for(int qn=0;qn<2;++qn){int n=wi*2+qn;uint8_t* sw=sm+131072+n*16384;
  float acc[16]={};fence_regs(acc);wgmma_fence();
  static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(acc,smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+212992+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });
  // Exact same K order and BF16 rounding, with transposed operand ownership.
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);
#pragma unroll
  for(int q=0;q<4;++q){int c=n*64+w*16+lane/4,r=q*8+2*(lane%4);
   put32(sn,c,r,pack_bf16(acc[4*q],acc[4*q+1]));
   put32(sn,c+8,r,pack_bf16(acc[4*q+2],acc[4*q+3]));
  }
 }
 // All channel groups must finish dnorm stores before the LN row reduction.
 allsync();
 int row=2*(threadIdx.x%16),cg=threadIdx.x/16;
 float mu0=mus[row],mu1=mus[row+1],rs0=rss[row],rs1=rss[row+1];
 float s10=0,s11=0,s20=0,s21=0;
#pragma unroll 1
 for(int k=0;k<16;++k){int c=cg*16+k;uint32_t x=get32(sx,c,row),dn=get32(sn,c,row);float g=gam[c];
  float xa=__fmul_rn(__fsub_rn(bf16lo(x),mu0),rs0),xb=__fmul_rn(__fsub_rn(bf16hi(x),mu1),rs1);
  float ha=bf16lo(dn)*g,hb=bf16hi(dn)*g;
  s10+=ha*xa;s11+=hb*xb;s20+=ha;s21+=hb;
 }
 // Each thread owns one row pair in one of16 disjoint channel groups.
 stats[cg*64+row*2]=s10;stats[cg*64+row*2+1]=s20;
 stats[cg*64+(row+1)*2]=s11;stats[cg*64+(row+1)*2+1]=s21;
 allsync(); // Publish all channel partials before their32 row owners reduce.
 if(threadIdx.x<32){int rr=threadIdx.x;float a=0,b=0;
#pragma unroll
  for(int k=0;k<16;++k){a+=stats[k*64+rr*2];b+=stats[k*64+rr*2+1];}
  // Overwrites only this same row's already-consumed partials. Other row
  // owners never read these addresses, so no extra intermediate barrier.
  stats[rr*2]=a/256.f;stats[rr*2+1]=b/256.f;
  stats[128+rr*2]=0;stats[128+rr*2+1]=0;
 }
 allsync(); // Publish combined row sums before any channel epilogue reads them.
 float* red=reinterpret_cast<float*>(sm+229376);
#pragma unroll 1
 for(int b=0;b<8;++b){int c=wi*128+b*16+w*4+lane/8;float dg=0,db=0,gamma=gam[c];
#pragma unroll
  for(int k=0;k<2;++k){int r=2*(lane%8)+16*k;uint32_t x=get32(sx,c,r),dn=get32(sn,c,r);float mua=mus[r],mub=mus[r+1],rsa=rss[r],rsb=rss[r+1];float xa=__fmul_rn(__fsub_rn(bf16lo(x),mua),rsa),xb=__fmul_rn(__fsub_rn(bf16hi(x),mub),rsb),da=bf16lo(dn),dd=bf16hi(dn);dg+=da*xa+dd*xb;db+=da+dd;put32(sx,c,r,pack_bf16(rsa*((da*gamma-(stats[r*2+1]+stats[128+r*2+1]))-xa*(stats[r*2]+stats[128+r*2])),rsb*((dd*gamma-(stats[(r+1)*2+1]+stats[128+(r+1)*2+1]))-xb*(stats[(r+1)*2]+stats[128+(r+1)*2]))));}
#pragma unroll
  for(int sh=1;sh<8;sh*=2){dg+=__shfl_xor_sync(0xffffffff,dg,sh);db+=__shfl_xor_sync(0xffffffff,db,sh);}
  if(lane%8==0){red[c]+=dg;red[256+c]+=db;}
 }
 // Generic writes visible to TMA; issuer waits before caller releases stage.
 sync_group();fence_proxy_async();sync_group();
 if(tid==0){for(int c=wi*128;c<(wi+1)*128;c+=16)tma_store_3d(&p.dtri,sx+c*64,m0,c,0);tma_store_commit();tma_store_wait_all();}
}
