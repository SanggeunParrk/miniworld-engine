from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_cluster.cu').read_text();h=s[:s.index('TMN_DEVI void producer')];h=h.replace('ready,empty[4],copy,io,weight[2]','ready,empty[4],copy,io,weight[2]')
body=r'''
extern "C" __global__ __cluster_dims__(8,1,1) __launch_bounds__(256,1) void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ Bars b;int rank=blockIdx.x%8;for(int i=threadIdx.x;i<131072/4;i+=256)reinterpret_cast<unsigned*>(sm)[i]=blockIdx.x;
 if(threadIdx.x==0){for(int q=0;q<4;++q)mbar_init(b.empty+q,1);mbar_init(&b.copy,1);if(rank>=4)mbar_arrive_expect_tx(&b.copy,131072);fence_barrier_init();}fence_proxy_async();allsync();auto c=cooperative_groups::this_cluster();c.sync();
 for(int r=0;r<128;++r){
  if(rank<4){if(threadIdx.x==0){for(int q=0;q<4;++q){if(r)mbar_wait(b.empty+q,(r-1)&1);cluster_copy(remote_addr(sm+rank*32768,4+q),sm,remote_addr(&b.copy,4+q));}}allsync();}
  else{mbar_wait(&b.copy,r&1);allsync();if(threadIdx.x==0){if(r+1<128)mbar_arrive_expect_tx(&b.copy,131072);for(int q=0;q<4;++q)remote_arrive(b.empty+rank-4,q);}}
 }
 c.sync();if(rank>=4&&threadIdx.x<4)p.partln[(blockIdx.x/8*4+rank-4)*256+threadIdx.x]=float(reinterpret_cast<unsigned*>(sm)[threadIdx.x*8192]);
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){}
''';(p/'front_cluster_copy_probe.cu').write_text(h+body);(p/'front_cluster_copy_probe.launch.json').write_text((p/'front_cluster.launch.json').read_text())
(p/'cluster_copy_probe.py').write_text('''from cluster_plan import *
with torch.no_grad():
 a=setup(64);p=ClusterPlan(a,120,'front_cluster_copy_probe');p();torch.cuda.synchronize();expected=(torch.arange(15,device='cuda')[:,None]*8+torch.arange(4,device='cuda')[None,:]).repeat_interleave(4,dim=0).float();assert torch.equal(p.partln[:,:4],expected)
 ts=paired({'copy_only':capture(p)});us=ts['copy_only']['median_us'];r=dict(median_us=us,bytes=15*4*4*32768*128,bandwidth_TBps=15*4*4*32768*128/us/1e6);(R/'cluster-copy-bandwidth.json').write_text(json.dumps(r,indent=2));print('DSM_COPY',r,flush=True)
''')
