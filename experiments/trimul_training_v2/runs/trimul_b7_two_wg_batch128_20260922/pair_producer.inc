// SPDX-License-Identifier: Apache-2.0
// Two independent row consumers share a CTA-local weight pipeline.
TMN_DEVI void pair_producer(const Params& p,uint8_t* sm,uint64_t* bar){
 int rank=2*(blockIdx.x%GROUP-SOURCES),cid=blockIdx.x/GROUP,groups=gridDim.x/GROUP;
 int warp=threadIdx.x/32,lane=threadIdx.x%32;
 if(warp==1||warp==2){int g=warp-1;
  for(int round=0,tile=cid+rank*groups;tile<p.tiles;tile+=CONSUMERS*groups,++round){
   if(round)mbar_wait(bar+18,(round-1)&1);
   int sequence=rank+g+round*CONSUMERS,rs=sequence%RINGS;
   bool valid=tile+g*groups<p.tiles;

   unsigned* flags=p.flags+(cid*RINGS+rs)*(SOURCES+2);
   for(int phase=0;phase<8;++phase){int slot=0,plane=phase/2,chunk_source=(phase%2)*4;
    if(phase>=1&&lane==0)mbar_wait(bar+10+g*4+slot,(phase-1)&1);
    __syncwarp();
    if(valid&&(plane&1)==0&&lane<4)ring_wait(flags-(sequence&1)*(SOURCES+2)+1+(plane/2)*8+chunk_source+lane,(sequence&~1)+1);
    __syncwarp();
    if(valid){
     if(lane==0){mbar_arrive_expect_tx(bar+8+g*4+slot,16384);bulk_load(sm+g*16384,p.ring+(cid*RINGS+rs)*131072+phase*16384,16384,bar+8+g*4+slot);}
    }else{
     for(int i=lane*16;i<16384;i+=512)*reinterpret_cast<uint4*>(sm+g*16384+i)=make_uint4(0,0,0,0);
     fence_proxy_async();__syncwarp();if(lane==0)mbar_arrive(bar+8+g*4+slot);
    }
   }
   if(lane==0){mbar_wait(bar+8+g*4,1);if(valid)publish(flags+SOURCES+1,sequence+1);}
   __syncwarp();
  }
  return;
 }
 if(threadIdx.x)return;
 for(int round=0,tile=cid+rank*groups;tile<p.tiles;tile+=CONSUMERS*groups,++round){
  if(round)mbar_wait(bar+18,(round-1)&1);
  for(int batch=0;batch<8;++batch){int ws=batch%2;
   if(batch>=2)mbar_wait(bar+2+ws,((batch/2)-1)&1);
   mbar_arrive_expect_tx(bar+ws,32768);
   for(int sub=0;sub<2;++sub){int stage=batch*2+sub;tma_load_2d(sm+32768+ws*32768+sub*16384,p.wt+stage/4,bar+ws,(stage%4)*64,0);}
  }
  // Both WGMMA consumers have completed; ring/weight storage is now reusable.
  for(int ws=0;ws<2;++ws)mbar_wait(bar+2+ws,1);
  mbar_arrive_expect_tx(bar+19,98304);
  for(int g=0;g<2;++g)for(int k=0;k<2;++k){
   tma_load_2d(sm+g*32768+k*8192,&p.dg,bar+19,k*64,(tile+g*groups)*64);
   tma_load_2d(sm+g*32768+16384+k*8192,&p.x,bar+19,k*64,(tile+g*groups)*64);
  }
  for(int k=0;k<2;++k)tma_load_2d(sm+65536+k*16384,&p.wgate,bar+19,k*64,0);
 }
}
