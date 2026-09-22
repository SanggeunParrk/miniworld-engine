from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_twocta_maskrcp.cu').read_text();a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a)
body=r'''
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b){
 int group=blockIdx.x/DW_SPLITS,split=blockIdx.x%DW_SPLITS,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2,r=0;float acc[64]={};
 if(split<p.tiles)load_dw(p,sm,b,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw(p,sm,b,1,(split+DW_SPLITS)*64,group);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r&1;uint8_t* s=sm+slot*DW_SLOT;mbar_wait(b+slot,(r/2)&1);glu_small(p,s,s+40960,s+49152,tile*64);
  fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_weight128(acc,smem_desc(smem_u32(s+40960+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(s+24576+k*2048),8192,1024,1),r%segmentRounds>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
  if(tile+2*DW_SPLITS<p.tiles)load_dw(p,sm,b,slot,(tile+2*DW_SPLITS)*64,group);
  if((r+1)%segmentRounds==0||tile+DW_SPLITS>=p.tiles){float* out=p.partw+((group*DW_SPLITS+split)*2+r/segmentRounds)*16384+wi*8192;
   static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,acc[q*4],acc[q*4+1]);stg64f(out+(rr+8)*128+c,acc[q*4+2],acc[q*4+3]);});}
 }
}
'''
s=s[:a]+body+s[b:];(p/'front_twocta_kindwg.cu').write_text(s);(p/'front_twocta_kindwg.launch.json').write_text((p/'front_twocta.launch.json').read_text())
ov=body.replace('mbar_wait(b+slot,(r/2)&1);glu_small(p,s,s+40960,s+49152,tile*64);','if(r==0){mbar_wait(b,(r/2)&1);glu_small(p,s,s+40960,s+49152,tile*64);}')
ov=ov.replace('});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();','''});wgmma_commit();
  if(tile+DW_SPLITS<p.tiles){int ns=slot^1;uint8_t* next=sm+ns*DW_SLOT;mbar_wait(b+ns,((r+1)/2)&1);glu_small(p,next,next+40960,next+49152,(tile+DW_SPLITS)*64);}
  wgmma_wait<0>();fence_regs(acc);allsync();''')
(p/'front_twocta_kindover.cu').write_text(s[:a]+ov+s[a+len(body):]);(p/'front_twocta_kindover.launch.json').write_text((p/'front_twocta.launch.json').read_text())
