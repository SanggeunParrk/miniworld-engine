// SPDX-License-Identifier: Apache-2.0
TMN_DEVI void single_producer(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x)return;int round=0;
 for(int tile=blockIdx.x;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64;
  if(!NEXTROW || round==0){mbar_arrive_expect_tx(bar+4,16384);for(int c=0;c<2;++c)tma_load_2d(sm+(NEXTROW?98304:81920)+c*8192,&p.xn,bar+4,c*64,row);}
  for(int seq=0;seq<16;++seq){int side=seq/8,h=seq%4,slot=h&1;
   if(seq>=2)mbar_wait(bar+8+slot,((seq/2)-1)&1);
   if(seq%8<4){if(!(NEXTROW>=2 && round>0 && seq==0))load_g(p,sm,bar,row,side,h);}else load_p(p,sm,bar,side,h);
   if(seq==15)load_ln_next(p,sm+65536,bar+3,row);
  }
  mbar_wait(bar+8,1);mbar_wait(bar+9,1);issue_gate(p,sm,bar+2,row);
  #if NEXTROW
  if(tile+DXCOUNT<p.tiles){int next=(tile+DXCOUNT)*64;mbar_arrive_expect_tx(bar+4,16384);for(int c=0;c<2;++c)tma_load_2d(sm+98304+c*8192,&p.xn,bar+4,c*64,next);}
#endif
#if NEXTROW >= 2
  mbar_wait(bar+6,round&1);if(tile+DXCOUNT<p.tiles)load_g(p,sm,bar,(tile+DXCOUNT)*64,0,0);
#endif
  mbar_wait(bar+5,round&1);
 }
}
