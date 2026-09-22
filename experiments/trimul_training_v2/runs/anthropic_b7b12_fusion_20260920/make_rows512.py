from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'front_rows128_p48c224.cu').read_text()
# Keep the proven dW pipeline, reserve the fourth WG there.
s=s.replace('TMN_DEVI int dxid(){return int(threadIdx.x)-128;}','TMN_DEVI int dxid(){return int(threadIdx.x)-256;}')
# A producer WG owns GLU; these loads have one issuing thread.
for name in ['load_gate128','load_glu128','load_pw','load_ln128']:
 a=s.index('TMN_DEVI void '+name);b=s.index('{',a);s=s[:b+1]+'\n if(threadIdx.x)return;'+s[b+1:]
a=s.index('TMN_DEVI void glu_rows128');b=s.index('TMN_DEVI void dx_consumer',a)
glu=s[a:b].replace('int tid=dxid();','int tid=threadIdx.x;').replace('dxsync();','psync();')
s=s[:a]+s[b:];a=s.index('TMN_DEVI void dx_producer');b=s.index('TMN_DEVI void dx_consumer',a)
prod=r'''
TMN_DEVI void dx_producer(const Params& p,uint8_t* sm,Barriers* b){
 int split=blockIdx.x-DWCOUNT,round=0,tiles=p.tiles/2;
 for(int tile=split;tile<tiles;tile+=DXCOUNT,++round){int row=tile*128,ph=round&1;
  if(round==0)load_gate128(p,sm,b,row);
  if(round)mbar_wait(b->empty,ph^1);load_glu128(p,sm,b,0,row,0);
  if(round)mbar_wait(b->empty+2,1);load_pw(p,sm,b,0,0);
  if(round)mbar_wait(b->empty+3,1);load_pw(p,sm,b,1,0);
  mbar_wait(b->empty+1,ph);load_glu128(p,sm,b,1,row,0);
  mbar_wait(b->tx,ph);glu_rows128(p,sm,0,row);ready(p,b,0);
  mbar_wait(b->tx+1,ph^1);glu_rows128(p,sm,1,row);ready(p,b,1);
  mbar_wait(b->empty,ph);load_glu128(p,sm,b,2,row,0);
  mbar_wait(b->empty+1,ph^1);load_glu128(p,sm,b,3,row,0);
  mbar_wait(b->tx,ph^1);glu_rows128(p,sm,2,row);ready(p,b,0);
  mbar_wait(b->tx+1,ph);glu_rows128(p,sm,3,row);ready(p,b,1);
  mbar_wait(b->empty,ph^1);load_glu128(p,sm,b,0,row,1);
  mbar_wait(b->empty+1,ph);load_glu128(p,sm,b,1,row,1);
  mbar_wait(b->empty+2,0);load_pw(p,sm,b,2,0);
  mbar_wait(b->empty+3,0);load_pw(p,sm,b,3,0);
  mbar_wait(b->empty+2,1);load_pw(p,sm,b,0,1);
  mbar_wait(b->empty+3,1);load_pw(p,sm,b,1,1);
  mbar_wait(b->tx,ph);glu_rows128(p,sm,0,row);ready(p,b,0);
  mbar_wait(b->tx+1,ph^1);glu_rows128(p,sm,1,row);ready(p,b,1);
  mbar_wait(b->empty,ph);load_glu128(p,sm,b,2,row,1);
  mbar_wait(b->empty+1,ph^1);load_glu128(p,sm,b,3,row,1);
  mbar_wait(b->tx,ph^1);glu_rows128(p,sm,2,row);ready(p,b,0);
  mbar_wait(b->tx+1,ph);glu_rows128(p,sm,3,row);ready(p,b,1);
  mbar_wait(b->empty,ph^1);load_ln128(p,sm,b,row);
  mbar_wait(b->empty+1,ph);if(tile+DXCOUNT<tiles)load_gate128(p,sm,b,(tile+DXCOUNT)*128);
  mbar_wait(b->empty+2,0);load_pw(p,sm,b,2,1);
  mbar_wait(b->empty+3,0);load_pw(p,sm,b,3,1);
 }
}
'''
s=s[:a]+glu+prod+s[b:]
s=s.replace('mbar_wait(b->tx+slot,ph^(((h+1)/2)&1));glu_rows128(p,sm,h,row);','mbar_wait(b->ready+slot,h/2);')
s=s.replace('__launch_bounds__(384,1)','__launch_bounds__(512,1)').replace('named_bar_sync(0,384)','named_bar_sync(0,512)')
s=s.replace('if(wg<2){setmaxnreg_dec<128>();dw_producer(p,sm,&bars);}else{setmaxnreg_inc<240>();dw_consumer(p,sm,&bars);}', 'if(wg<2){setmaxnreg_dec<120>();dw_producer(p,sm,&bars);}else if(wg==2){setmaxnreg_inc<240>();dw_consumer(p,sm,&bars);}else{setmaxnreg_dec<24>();}')
s=s.replace('if(wg==0){setmaxnreg_dec<48>();dx_producer(p,sm,&bars);}else{setmaxnreg_inc<224>();dx_consumer(p,sm,&bars,gamma);}', 'if(wg<2){setmaxnreg_dec<64>();dx_producer(p,sm,&bars);}else{setmaxnreg_inc<192>();dx_consumer(p,sm,&bars,gamma);}')
s=s.replace('blockIdx.x*384+threadIdx.x','blockIdx.x*512+threadIdx.x').replace('i+=UCOUNT*384','i+=UCOUNT*512')
(p/'front_rows512.cu').write_text(s);(p/'front_rows512.launch.json').write_text('{"wgrad_slices":2,"threads":512}\n')
