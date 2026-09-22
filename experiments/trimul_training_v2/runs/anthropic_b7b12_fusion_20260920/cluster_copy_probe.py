from cluster_plan import *
with torch.no_grad():
 a=setup(64);p=ClusterPlan(a,120,'front_cluster_copy_probe');p();torch.cuda.synchronize();expected=(torch.arange(15,device='cuda')[:,None]*8+torch.arange(4,device='cuda')[None,:]).repeat_interleave(4,dim=0).float();assert torch.equal(p.partln[:,:4],expected)
 ts=paired({'copy_only':capture(p)});us=ts['copy_only']['median_us'];r=dict(median_us=us,bytes=15*4*4*32768*128,bandwidth_TBps=15*4*4*32768*128/us/1e6);(R/'cluster-copy-bandwidth.json').write_text(json.dumps(r,indent=2));print('DSM_COPY',r,flush=True)
