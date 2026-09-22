from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_cluster.cu').read_text();s=s.replace('ready,empty[4],copy,io,weight[2]','ready,empty[4],free[4],copy,io[2],weight[2]')
a=s.index('TMN_DEVI void producer');b=s.index('TMN_DEVI void load_weight',a);old=s[a:b];body=old[old.index('   uint32_t mask'):];suffix='\n  }\n }\n}\n';assert body.endswith(suffix);body=body[:-len(suffix)]
body=body.replace('#pragma unroll 2','#pragma unroll 4')
new=r'''TMN_DEVI void load_cluster_input(const Params& p,uint8_t* sm,Bars* b,int r,int total){
 if(threadIdx.x||r>=total)return;int cid=blockIdx.x/8,group=blockIdx.x%8,row=(cid*4+(r%4)+(r/4)*CLUSTERS*4)*64,slot=r&1;uint8_t* inp=sm+65536+slot*65536;uint64_t* io=b->io+slot;
 mbar_arrive_expect_tx(io,65536);tma_load_2d(inp,&p.pre,io,row,(group/2)*512+(group%2)*256);tma_load_2d(inp+32768,group>=2?&p.dr:&p.dl,io,row,(group%2)*128);for(int c=0;c<2;++c)tma_load_2d(inp+49152+c*8192,&p.xn,io,c*64,row);
}
TMN_DEVI void producer(const Params& p,uint8_t* sm,Bars* b){
 int cid=blockIdx.x/8,group=blockIdx.x%8,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 float acc[2][64]={};int total=0;for(int base=cid*4;base<p.tiles;base+=CLUSTERS*4)total+=min(4,p.tiles-base);int segment=(total+1)/2;
 load_cluster_input(p,sm,b,0,total);load_cluster_input(p,sm,b,1,total);
 for(int r=0;r<total;++r){int q=r%4,round=r/4,row=(cid*4+q+round*CLUSTERS*4)*64;uint8_t* out=sm+(q%2)*32768;uint8_t* inp=sm+65536+(r%2)*65536;
  if(threadIdx.x==0){if(r>=2)mbar_wait(b->empty+q%2,(r/2-1)&1);if(round)mbar_wait(b->free+q,(round-1)&1);}mbar_wait(b->io+(r%2),(r/2)&1);allsync();
'''+body+r'''
  allsync();load_cluster_input(p,sm,b,r+2,total);
 }
}
'''
s=s[:a]+new+s[b:];s=s.replace('uint64_t* bar=&b->io','uint64_t* bar=b->io')
s=s.replace('mbar_wait(&b->copy,round&1);','mbar_wait(&b->copy,round&1);if(threadIdx.x==0)for(int group=0;group<4;++group)remote_arrive(b->empty+slot%2,group);')
s=s.replace('remote_arrive(b->empty+slot,group);','remote_arrive(b->free+slot,group);')
s=s.replace('for(int i=0;i<4;++i)mbar_init(b.empty+i,1);','for(int i=0;i<4;++i){mbar_init(b.empty+i,1);mbar_init(b.free+i,1);}')
s=s.replace('mbar_init(&b.io,1);','mbar_init(b.io,1);mbar_init(b.io+1,1);')
(p/'front_cluster_pipe.cu').write_text(s);(p/'front_cluster_pipe.launch.json').write_text((p/'front_cluster.launch.json').read_text())
