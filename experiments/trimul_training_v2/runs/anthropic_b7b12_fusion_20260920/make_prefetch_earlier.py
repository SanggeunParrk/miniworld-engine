from pathlib import Path
p=Path(__file__).resolve().parent;src=(p/'front_prefetch_lnpair.cu').read_text()
for dw in [False,True]:
 for dx in [False,True]:
  if not dw and not dx:continue
  s=src
  if dw:
   a=s.index('TMN_DEVI void weight_role')
   helper='''TMN_DEVI void load_dw_data(const Params& p,uint8_t* s,uint64_t* b,int row,int group){if(threadIdx.x)return;int side=group/4,h=(group%4)*64;mbar_arrive_expect_tx(b,40960);tma_load_2d(s,&p.pre,b,row,side*512+h*2);tma_load_2d(s+16384,side?&p.dr:&p.dl,b,row,h);}
TMN_DEVI void load_dw_xn(const Params& p,uint8_t* s,uint64_t* b,int row){if(threadIdx.x)return;for(int n=0;n<2;++n)tma_load_2d(s+24576+n*8192,&p.xn,b,n*64,row);}
''';s=s[:a]+helper+s[a:]
   s=s.replace('glu_small(p,s,s+40960,s+49152,tile*64);','glu_small(p,s,s+40960,s+49152,tile*64);if(tile+2*DW_SPLITS<p.tiles)load_dw_data(p,s,b+slot,(tile+2*DW_SPLITS)*64,group);')
   s=s.replace('load_dw(p,sm,b,slot,(tile+2*DW_SPLITS)*64,group);','load_dw_xn(p,s,b+slot,(tile+2*DW_SPLITS)*64);')
  if dx:
   a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
   v=v.replace('load_g(p,sm,bar,row,side,0);load_g(p,sm,bar,row,side,1);','if(side==0)load_g(p,sm,bar,row,side,0);load_g(p,sm,bar,row,side,1);')
   v=v.replace('if(h<2)load_g(p,sm,bar,row,side,h+2);','if(h<2)load_g(p,sm,bar,row,side,h+2);if(h==2)load_p(p,sm,bar,side,0);')
   v=v.replace('load_p(p,sm,bar,side,0);load_p(p,sm,bar,side,1);','load_p(p,sm,bar,side,1);')
   v=v.replace('if(side==1&&h==1)load_ln_next(p,sm+65536,bar+3,row);','if(side==1&&h==1)load_ln_next(p,sm+65536,bar+3,row);if(side==0&&h==2)load_g(p,sm,bar,row,1,0);')
   s=s[:a]+v+s[b:]
  name='front_prefetch_early'+('dw' if dw else '')+('dx' if dx else '');(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
