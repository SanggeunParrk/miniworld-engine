from pathlib import Path
p=Path(__file__).resolve().parent;src=(p/'front_prefetch_earlydw.cu').read_text()
for early in [False,True]:
 s=src;a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a)
 body=r'''TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 int group=blockIdx.x%8,split=blockIdx.x/8,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,batches=(rounds+1)/2,seg=(batches+1)/2,round=0;float acc[64]={};
 if(split<p.tiles)load_dw(p,sm,b,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw(p,sm,b,1,(split+DW_SPLITS)*64,group);
 for(int tile=split;tile<p.tiles;tile+=2*DW_SPLITS,++round){
  mbar_wait(b,round&1);glu_small(p,sm,sm+40960,sm+49152,tile*64);
  if(tile+DW_SPLITS<p.tiles){mbar_wait(b+1,round&1);glu_small(p,sm+DW_SLOT,sm+DW_SLOT+40960,sm+DW_SLOT+49152,(tile+DW_SPLITS)*64);}
  EARLY_DATA
  fence_regs(acc);wgmma_fence();
  #pragma unroll 1
  for(int half=0;half<2;++half){if(tile+half*DW_SPLITS<p.tiles){uint8_t* s=sm+half*DW_SLOT;static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_weight128(acc,smem_desc(smem_u32(s+40960+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(s+24576+k*2048),8192,1024,1),round%seg>0||half>0||k>0);});}}
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
  REFILL
  if((round+1)%seg==0||tile+2*DW_SPLITS>=p.tiles){float* out=p.partw+((group*DW_SPLITS+split)*2+round/seg)*16384+wi*8192;static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[q*4],acc[q*4+1]);stg64f(out+(rr+8)*128+c,acc[q*4+2],acc[q*4+3]);});}
 }
}
'''
 if early:
  body=body.replace('EARLY_DATA','for(int half=0;half<2;++half)if(tile+(half+2)*DW_SPLITS<p.tiles)load_dw_data(p,sm+half*DW_SLOT,b+half,(tile+(half+2)*DW_SPLITS)*64,group);').replace('REFILL','for(int half=0;half<2;++half)if(tile+(half+2)*DW_SPLITS<p.tiles)load_dw_xn(p,sm+half*DW_SLOT,b+half,(tile+(half+2)*DW_SPLITS)*64);')
 else:body=body.replace('EARLY_DATA','').replace('REFILL','for(int half=0;half<2;++half)if(tile+(half+2)*DW_SPLITS<p.tiles)load_dw(p,sm,b,half,(tile+(half+2)*DW_SPLITS)*64,group);')
 s=s[:a]+body+s[b:];name='front_prefetch_dwbatch'+('_early' if early else '');(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
