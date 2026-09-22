// Recreate only one 64-hidden-channel gate/projection pair on chip.
TMN_DEVI void recompute_pre(uint8_t* pre,uint8_t* xn,uint8_t* wg,uint8_t* wp){
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 float acc[32];recompute_gemm<128,8192>(acc,xn,wi==0?wg:wp);
 // All weight reads retire before their shared memory can become preactivation.
 allsync();
#pragma unroll
 for(int q=0;q<4;++q){int mat=lane/8,chq=lane%8+8*(mat>>1);
  stsm_x4_t(smem_u32(pre+wi*8192)+swz128(q*16+chq,(w*16+8*(mat&1))*2),
   pack_bf16(acc[q*8],acc[q*8+1]),pack_bf16(acc[q*8+2],acc[q*8+3]),pack_bf16(acc[q*8+4],acc[q*8+5]),pack_bf16(acc[q*8+6],acc[q*8+7]));
 }
 allsync();
}
TMN_DEVI void recompute_dw(const Params& p,uint8_t* s,const float* gamma,const float* beta){
 if(threadIdx.x<128)normalize_tile<8>(s+24576,s+24576,gamma,beta);
 fence_proxy_async();allsync();
 recompute_pre(s,s+24576,s,s+40960);
}
TMN_DEVI void load_dw(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){
 if(threadIdx.x)return;uint8_t* s=sm+slot*DW_SLOT;int side=group/4,h=(group%4)*64;mbar_arrive_expect_tx(b+slot,57344);
 tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h);
 for(int n=0;n<2;++n){
  tma_load_2d(s+n*8192,side?&p.wrg:&p.wlg,b+slot,h,n*64);
  tma_load_2d(s+40960+n*8192,side?&p.wr:&p.wl,b+slot,h,n*64);
  tma_load_2d(s+24576+n*8192,&p.x,b+slot,n*64,row);
 }
}
