from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'front_warp.cu').read_text();s=s.replace('DX_SLOT=57344','DX_SLOT=114688')
old=(p/'front_gamma.cu').read_text()
# Existing two-consumer GLU/epilogue, with a separate TMA producer.
glu=old[old.index('// Each pair'):old.index('TMN_DEVI void issue_dw')]
glu=glu.replace('glu_pair','glu128_pair').replace('write_glu','write_glu128').replace('void glu(','void glu128(').replace('threadIdx.x','dxid()').replace('allsync()','dxsync()')
prefix='''TMN_DEVI int dxid(){return int(threadIdx.x)-128;}
TMN_DEVI void dxsync(){named_bar_sync(2,256);}
TMN_DEVI void dxrelease(Barriers* b,int slot){fence_proxy_async();dxsync();if(dxid()==0)mbar_arrive(b->empty+slot);}
'''+glu+'''
TMN_DEVI void h_load_gate(const Params& p,uint8_t* sm,Barriers* b,int slot,int row){
 mbar_arrive_expect_tx(b->tx+slot,49152);
 for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.dg,b->tx+slot,k*64,row);
  for(int n=0;n<2;++n)tma_load_2d(sm+16384+n*16384+k*8192,&p.wgate,b->tx+slot,k*64,n*64);}
}
TMN_DEVI void h_load_front(const Params& p,uint8_t* sm,Barriers* b,int slot,int row,int group){
 uint8_t* s=sm+slot*DX_SLOT;int side=group/2,h=(group%2)*128;
 mbar_arrive_expect_tx(b->tx+slot,114688);
 for(int c=0;c<2;++c){tma_load_2d(s+c*16384,&p.pre,b->tx+slot,row,side*512+2*h+c*128);tma_load_2d(s+32768+c*8192,side?&p.dr:&p.dl,b->tx+slot,row,h+c*64);}
 for(int kind=0;kind<2;++kind)for(int n=0;n<2;++n)for(int k=0;k<2;++k)tma_load_2d(s+49152+kind*32768+n*16384+k*8192,side?(kind?&p.wr:&p.wrg):(kind?&p.wl:&p.wlg),b->tx+slot,h+k*64,n*64);
}
TMN_DEVI void dx_producer(const Params& p,uint8_t* sm,Barriers* b){
 if(threadIdx.x)return;int split=blockIdx.x-DWCOUNT,round=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64,a=round&1,other=a^1;
  if(round)mbar_wait(b->empty+a,1);h_load_gate(p,sm+a*DX_SLOT,b,a,row);
  if(round)mbar_wait(b->empty+other,1);h_load_front(p,sm,b,other,row,1);
  mbar_wait(b->empty+a,0);h_load_front(p,sm,b,a,row,0);
  mbar_wait(b->empty+a,1);h_load_front(p,sm,b,a,row,2);
  mbar_wait(b->empty+other,0);h_load_front(p,sm,b,other,row,3);
  mbar_wait(b->empty+a,0);mbar_arrive_expect_tx(b->tx+a,32768);
  for(int c=0;c<2;++c){tma_load_2d(sm+a*DX_SLOT+c*8192,&p.x,b->tx+a,c*64,row);tma_load_2d(sm+a*DX_SLOT+16384+c*8192,&p.res,b->tx+a,c*64,row);}
 }
}
'''
body=old[old.index('TMN_DEVI void input_role'):old.index('TMN_DEVI void reduce_at')]
body=body.replace('input_role','dx_consumer').replace('uint64_t* bar','Barriers* b').replace('threadIdx.x','dxid()').replace('allsync()','dxsync()')
body=body.replace('float running=0;','float running=0;int round=0;').replace('tile+=DXCOUNT){','tile+=DXCOUNT,++round){').replace('int row=tile*64;','int row=tile*64,a=round&1;uint8_t* gs=sm+a*DX_SLOT;')
body=body.replace('issue_gate(p,sm,bar,row);mbar_wait(bar,0);','mbar_wait(b->tx+a,0);')
ba=body.index('{float gate[32]');be=body.index('dxsync();issue_dx',ba)
body=body[:ba]+body[ba:be].replace('sm+','gs+')+body[be:]
body=body.replace('dxsync();issue_dx(p,sm,bar,0,row,0);issue_dx(p,sm,bar,1,row,1);','dxrelease(b,a);')
body=body.replace('sm+slot*DX_SLOT','sm+(slot^a)*DX_SLOT').replace('sm+half*DX_SLOT','sm+(half^a)*DX_SLOT')
body=body.replace('mbar_wait(bar+slot,(side+1-half)&1);glu<true>','mbar_wait(b->tx+(slot^a),(side+1-half)&1);glu128<true>')
body=body.replace('if(side==0)issue_dx(p,sm,bar,half,row,2+half);','dxrelease(b,half^a);')
ba=body.index('  if(dxid()==0){mbar_arrive_expect_tx')
be=body.index('  float mu[2]',ba)
body=body[:ba]+'  mbar_wait(b->tx+a,1);uint8_t* ln=sm+a*DX_SLOT;\n'+body[be:]
ba=body.index('  float mu[2]');be=body.index('\n p.partln',ba)
chunk=body[ba:be].replace('sm+','ln+').replace('store2d(&p.dx,sm,','store2d(&p.dx,ln,')
chunk=chunk.replace('dxsync(); // TMA store cannot','dxrelease(b,a); // TMA store cannot')
body=body[:ba]+chunk+body[be:]
start=s.index('TMN_DEVI void load_gate');end=s.index('TMN_DEVI void reduce_at',start)
s=s[:start]+prefix+body+s[end:]
s=s.replace('if(wg<2){setmaxnreg_dec<128>();if(blockIdx.x<DWCOUNT)dw_producer(p,sm,&bars);else dx_producer(p,sm,&bars);}\n else{setmaxnreg_inc<240>();if(blockIdx.x<DWCOUNT)dw_consumer(p,sm,&bars);else dx_consumer(p,sm,&bars,gamma);}', '''if(blockIdx.x<DWCOUNT){
  if(wg<2){setmaxnreg_dec<128>();dw_producer(p,sm,&bars);}else{setmaxnreg_inc<240>();dw_consumer(p,sm,&bars);}
 }else{
  if(wg==0){setmaxnreg_dec<24>();dx_producer(p,sm,&bars);}else{setmaxnreg_inc<240>();dx_consumer(p,sm,&bars,gamma);}
 }''')
assert 'issue_dx(' not in s and 'issue_gate(' not in s
(p/'front_hybrid.cu').write_text(s)
